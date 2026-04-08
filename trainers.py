import torch

torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig
import gradient_attribution  # Import gradient attribution module

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple


def tdpo_loss(chosen_logps_margin: torch.FloatTensor,
              rejected_logps_margin: torch.FloatTensor,
              chosen_position_kl: torch.FloatTensor,
              rejected_position_kl: torch.FloatTensor,
              beta: float, alpha: float = 0.5, if_tdpo2: bool = True) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the TDPO loss for a batch of policy and reference model log probabilities.

    Args:
        chosen_logps_margin: The difference of log probabilities between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
        rejected_logps_margin: The difference of log probabilities between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
        chosen_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the chosen responses. Shape: (batch_size,)
        rejected_position_kl: The difference of sequential kl divergence between the policy model and the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the TDPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        alpha: Temperature parameter for the TDPO loss, used to adjust the impact of sequential kl divergence.
        if_tdpo2: Determine whether to use method TDPO2, default is True; if False, then use method TDPO1.

    Returns:
        A tuple of two tensors: (losses, rewards).
        The losses tensor contains the TDPO loss for each example in the batch.
        The rewards tensors contain the rewards for response pair.
    """
    
    # Check if inputs contain NaN or Inf
    if torch.isnan(chosen_logps_margin).any() or torch.isinf(chosen_logps_margin).any():
        print("Warning: chosen_logps_margin contains NaN or Inf")
        return torch.zeros(chosen_logps_margin.shape[0], device=chosen_logps_margin.device), \
               torch.zeros(chosen_logps_margin.shape[0], device=chosen_logps_margin.device), \
               torch.zeros(chosen_logps_margin.shape[0], device=chosen_logps_margin.device)
    
    if torch.isnan(rejected_logps_margin).any() or torch.isinf(rejected_logps_margin).any():
        print("Warning: rejected_logps_margin contains NaN or Inf")
        return torch.zeros(rejected_logps_margin.shape[0], device=rejected_logps_margin.device), \
               torch.zeros(rejected_logps_margin.shape[0], device=rejected_logps_margin.device), \
               torch.zeros(rejected_logps_margin.shape[0], device=rejected_logps_margin.device)

    chosen_values = chosen_logps_margin + chosen_position_kl
    rejected_values = rejected_logps_margin + rejected_position_kl

    chosen_rejected_logps_margin = chosen_logps_margin - rejected_logps_margin

    if not if_tdpo2:
        logits = chosen_rejected_logps_margin - (rejected_position_kl - chosen_position_kl)    # tdpo1
    else:
        logits = chosen_rejected_logps_margin - alpha * (rejected_position_kl - chosen_position_kl.detach())  # tdpo2
    
    # Check if logits contain NaN or Inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("Warning: logits contains NaN or Inf")
        return torch.zeros(logits.shape[0], device=logits.device), \
               torch.zeros(logits.shape[0], device=logits.device), \
               torch.zeros(logits.shape[0], device=logits.device)
    
    # Use more stable logsigmoid calculation
    try:
        losses = -F.logsigmoid(beta * logits)
    except Exception as e:
        print(f"logsigmoid calculation failed: {e}")
        # Use more stable method
        logits_clamped = torch.clamp(beta * logits, min=-10, max=10)
        losses = -F.logsigmoid(logits_clamped)

    chosen_rewards = beta * chosen_values.detach()
    rejected_rewards = beta * rejected_values.detach()

    return losses, chosen_rewards, rejected_rewards


def dpo_loss(chosen_logps_margin: torch.FloatTensor,
             rejected_logps_margin: torch.FloatTensor,
             beta: float) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the (weighted) DPO loss.

    For TI-DPO we use a token-importance-weighted log-ratio sum per sequence (Eq.10/11),
    then apply the standard DPO objective.
    """
    if torch.isnan(chosen_logps_margin).any() or torch.isinf(chosen_logps_margin).any():
        print("Warning: chosen_logps_margin contains NaN or Inf")
        zeros = torch.zeros(chosen_logps_margin.shape[0], device=chosen_logps_margin.device)
        return zeros, zeros, zeros

    if torch.isnan(rejected_logps_margin).any() or torch.isinf(rejected_logps_margin).any():
        print("Warning: rejected_logps_margin contains NaN or Inf")
        zeros = torch.zeros(rejected_logps_margin.shape[0], device=rejected_logps_margin.device)
        return zeros, zeros, zeros

    logits = chosen_logps_margin - rejected_logps_margin
    losses = -F.logsigmoid(beta * logits)
    chosen_rewards = beta * chosen_logps_margin.detach()
    rejected_rewards = beta * rejected_logps_margin.detach()
    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor,
                     average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

  Args:
      logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
      labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
      average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def _tdpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor,
                          average_log_prob: bool = False):
    """Compute the kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    # Check if inputs contain NaN or Inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("Warning: logits contains NaN or Inf values")
        batch_size = logits.shape[0]
        device = logits.device
        return torch.zeros(batch_size, device=device), \
               torch.zeros(batch_size, device=device), \
               torch.zeros(batch_size, device=device)
    
    if torch.isnan(reference_logits).any() or torch.isinf(reference_logits).any():
        print("Warning: reference_logits contains NaN or Inf values")
        batch_size = reference_logits.shape[0]
        device = reference_logits.device
        return torch.zeros(batch_size, device=device), \
               torch.zeros(batch_size, device=device), \
               torch.zeros(batch_size, device=device)

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    # Check if calculation results contain NaN or Inf
    if torch.isnan(logps_margin).any() or torch.isinf(logps_margin).any():
        print("Warning: logps_margin contains NaN or Inf values")
        batch_size = logps_margin.shape[0]
        device = logits.device
        return torch.zeros(batch_size, device=device), \
               torch.zeros(batch_size, device=device), \
               torch.zeros(batch_size, device=device)

    if average_log_prob:
        return (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * loss_mask).sum(-1), \
            (per_position_kl * loss_mask).sum(-1), \
            (per_token_logps * loss_mask).sum(-1)


def _weighted_tdpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor,
                                   weight_matrix: torch.FloatTensor, average_log_prob: bool = False):
    """Compute the weighted kl divergence/log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        reference_logits: Logits of the reference model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        weight_matrix: Token importance weights. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        Several tensors of shape (batch_size,) containing the weighted average/sum kl divergence/log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape
    assert weight_matrix.shape == labels.shape

    # Check if inputs contain NaN or Inf
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        print("Warning: logits contains NaN or Inf values")
        return torch.zeros(labels.shape[0], device=logits.device), \
               torch.zeros(labels.shape[0], device=logits.device), \
               torch.zeros(labels.shape[0], device=logits.device)
    
    if torch.isnan(reference_logits).any() or torch.isinf(reference_logits).any():
        print("Warning: reference_logits contains NaN or Inf values")
        return torch.zeros(labels.shape[0], device=logits.device), \
               torch.zeros(labels.shape[0], device=logits.device), \
               torch.zeros(labels.shape[0], device=logits.device)

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    weight_matrix = weight_matrix[:, 1:]  # Adjust weight matrix shape

    # Check if weight matrix contains NaN or Inf
    if torch.isnan(weight_matrix).any() or torch.isinf(weight_matrix).any():
        print("Warning: weight_matrix contains NaN or Inf values, using uniform weights")
        weight_matrix = torch.ones_like(weight_matrix)

    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)

    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()

    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    # Check if logps_margin contains NaN or Inf
    if torch.isnan(logps_margin).any() or torch.isinf(logps_margin).any():
        print("Warning: logps_margin contains NaN or Inf values")
        return torch.zeros(labels.shape[0], device=logits.device), \
               torch.zeros(labels.shape[0], device=logits.device), \
               torch.zeros(labels.shape[0], device=logits.device)

    # Apply token importance weights
    weighted_logps_margin = logps_margin * weight_matrix

    # Check if weighted_logps_margin contains NaN or Inf
    if torch.isnan(weighted_logps_margin).any() or torch.isinf(weighted_logps_margin).any():
        print("Warning: weighted_logps_margin contains NaN or Inf values")
        return torch.zeros(labels.shape[0], device=logits.device), \
               torch.zeros(labels.shape[0], device=logits.device), \
               torch.zeros(labels.shape[0], device=logits.device)

    if average_log_prob:
        return (weighted_logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_position_kl * loss_mask).sum(-1) / loss_mask.sum(-1), \
               (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (weighted_logps_margin * loss_mask).sum(-1), \
            (per_position_kl * loss_mask).sum(-1), \
            (per_token_logps * loss_mask).sum(-1)


def _dpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor,
                         average_log_prob: bool = False):
    """Compute (unweighted) log-ratio sums for DPO.

    Returns:
        logps_margin_sum: (B,)
        policy_logps_sum: (B,)
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_logps = reference_logits.log_softmax(-1)

    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    logps_margin = per_token_logps - per_reference_token_logps

    if average_log_prob:
        return (logps_margin * loss_mask).sum(-1) / loss_mask.sum(-1), (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (logps_margin * loss_mask).sum(-1), (per_token_logps * loss_mask).sum(-1)


def _position_kl_sum(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor) -> torch.FloatTensor:
    """Compute per-sequence sum of position-wise KL(reference || policy) over labeled tokens.

    This matches the per_position_kl used in TDPO, but returns only the sequence-level sum.
    Shape: (batch_size,)
    """
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_ps = reference_logits.softmax(-1)
    reference_vocab_logps = reference_vocab_ps.log()
    per_position_kl = (reference_vocab_ps * (reference_vocab_logps - vocab_logps)).sum(-1)
    return (per_position_kl * loss_mask).sum(-1)


def _weighted_dpo_get_batch_logps(logits: torch.FloatTensor, reference_logits: torch.FloatTensor, labels: torch.LongTensor,
                                  weight_matrix: torch.FloatTensor, average_log_prob: bool = False):
    """Compute token-importance-weighted log-ratio sums for DPO (TI-DPO Eq.10/11)."""
    assert logits.shape[:-1] == labels.shape
    assert reference_logits.shape[:-1] == labels.shape
    assert weight_matrix.shape == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    reference_logits = reference_logits[:, :-1, :]
    weight_matrix = weight_matrix[:, 1:]

    loss_mask = (labels != -100)
    labels[labels == -100] = 0

    vocab_logps = logits.log_softmax(-1)
    reference_vocab_logps = reference_logits.log_softmax(-1)

    per_token_logps = torch.gather(vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    per_reference_token_logps = torch.gather(reference_vocab_logps, dim=2, index=labels.unsqueeze(2)).squeeze(2)
    logps_margin = per_token_logps - per_reference_token_logps

    weighted_logps_margin = logps_margin * weight_matrix

    if average_log_prob:
        denom = (loss_mask.to(weighted_logps_margin.dtype) * weight_matrix).sum(-1).clamp_min(1e-8)
        return (weighted_logps_margin * loss_mask).sum(-1) / denom, (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (weighted_logps_margin * loss_mask).sum(-1), (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(
    batch: Dict[str, Union[List, torch.LongTensor]],
    pad_token_id: Optional[int] = None,
) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    # Check batch size to ensure it's not 0
    if batch['chosen_input_ids'].shape[0] == 0 or batch['rejected_input_ids'].shape[0] == 0:
        raise ValueError("Batch size cannot be 0. Please check your data and batch configuration.")
    
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])

    if pad_token_id is None:
        # Best-effort inference: if any padding exists, pick the most common token
        # where attention_mask == 0; otherwise fall back to 0.
        try:
            if (
                'chosen_attention_mask' in batch
                and isinstance(batch['chosen_attention_mask'], torch.Tensor)
                and isinstance(batch['chosen_input_ids'], torch.Tensor)
            ):
                pad_positions = (batch['chosen_attention_mask'] == 0)
                if pad_positions.any():
                    candidates = batch['chosen_input_ids'][pad_positions]
                    if candidates.numel() > 0:
                        values, counts = torch.unique(candidates, return_counts=True)
                        pad_token_id = int(values[counts.argmax()].item())
        except Exception:
            pad_token_id = None

    if pad_token_id is None:
        pad_token_id = 0
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            if 'labels' in k:
                pad_value = -100
            elif 'attention_mask' in k:
                pad_value = 0
            elif 'input_ids' in k:
                pad_value = int(pad_token_id)
            else:
                pad_value = 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            if 'labels' in k:
                pad_value = -100
            elif 'attention_mask' in k:
                pad_value = 0
            elif 'input_ids' in k:
                pad_value = int(pad_token_id)
            else:
                pad_value = 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str,
                 reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or TDPO training.

           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path,
                                                                    cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Decoder-only generation is more reliable with left padding.
        self.tokenizer.padding_side = 'left'

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy = policy
        self.reference_model = reference_model

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs,
                                                 n_examples=config.n_examples, batch_size=config.batch_size,
                                                 silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples,
                                                batch_size=config.eval_batch_size, silent=rank != 0,
                                                cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def _left_pad_for_generation(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Left-pad a batch for decoder-only generation.

        This is a runtime guard: even if upstream batching right-pads, we feed left-padded
        prompts to `generate()` to avoid degraded generation quality.
        """
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            raise ValueError(f"Expected 2D tensors, got input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
        if input_ids.shape != attention_mask.shape:
            raise ValueError(f"Shape mismatch: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")

        pad_token_id = int(self.tokenizer.pad_token_id)
        batch_size, seq_len = input_ids.shape
        out_ids = torch.full_like(input_ids, pad_token_id)
        out_mask = torch.zeros_like(attention_mask)
        lengths = attention_mask.to(torch.long).sum(dim=1)
        for i in range(batch_size):
            length_i = int(lengths[i].item())
            if length_i <= 0:
                continue
            tokens = input_ids[i][attention_mask[i].bool()]
            if tokens.numel() == 0:
                continue
            if tokens.numel() != length_i:
                length_i = int(tokens.numel())
            out_ids[i, seq_len - length_i:seq_len] = tokens[-length_i:]
            out_mask[i, seq_len - length_i:seq_len] = 1
        return out_ids, out_mask

    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing TDPO training) for the given batch of inputs."""

        prompt_input_ids, prompt_attention_mask = self._left_pad_for_generation(
            batch['prompt_input_ids'],
            batch['prompt_attention_mask'],
        )

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False,
                                               recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = self.policy.generate(
                prompt_input_ids, attention_mask=prompt_attention_mask,
                max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in ('tdpo', 'tidpo'):
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False,
                                                   recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    prompt_input_ids, attention_mask=prompt_attention_mask,
                    max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in ('tdpo', 'tidpo'):
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def tdpo_concatenated_forward(self, model: nn.Module, reference_model: nn.Module,
                                  batch: Dict[str, Union[List, torch.LongTensor]]):
        """Run the policy model and the reference model on the given batch of inputs, concatenating the chosen and rejected inputs together.

            We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        # Check batch size
        if batch['chosen_input_ids'].shape[0] == 0 or batch['rejected_input_ids'].shape[0] == 0:
            raise ValueError(f"Invalid batch size: chosen={batch['chosen_input_ids'].shape[0]}, rejected={batch['rejected_input_ids'].shape[0]}")
        
        concatenated_batch = concatenated_inputs(batch, pad_token_id=int(self.tokenizer.pad_token_id))
        
        # Check concatenated batch size
        if concatenated_batch['concatenated_input_ids'].shape[0] == 0:
            raise ValueError("Concatenated batch size is 0")
        
        # Optional expensive checks in debug mode
        if getattr(self.config, 'debug', False):
            if torch.isnan(concatenated_batch['concatenated_input_ids']).any():
                print("Warning: concatenated_input_ids contains NaN")
                raise ValueError("Input contains NaN")
        
        # Model forward pass
        try:
            all_logits = model(concatenated_batch['concatenated_input_ids'],
                               attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
            
            if getattr(self.config, 'debug', False):
                if torch.isnan(all_logits).any() or torch.isinf(all_logits).any():
                    print("Warning: Model output logits contain NaN or Inf")
                    print(f"logits shape: {all_logits.shape}")
                    print(f"logits range: [{all_logits.min()}, {all_logits.max()}]")
                    raise ValueError("Model output contains NaN or Inf")
                
        except Exception as e:
            print(f"Model forward pass failed: {e}")
            raise e
        
        with torch.no_grad():
            try:
                reference_all_logits = reference_model(concatenated_batch['concatenated_input_ids'],
                                                       attention_mask=concatenated_batch[
                                                           'concatenated_attention_mask']).logits.to(torch.float32)
                
                if getattr(self.config, 'debug', False):
                    if torch.isnan(reference_all_logits).any() or torch.isinf(reference_all_logits).any():
                        print("Warning: Reference model output logits contain NaN or Inf")
                        raise ValueError("Reference model output contains NaN or Inf")
                    
            except Exception as e:
                print(f"Reference model forward pass failed: {e}")
                raise e
        
        # Check if TIDPO (Token Importance DPO) is enabled
        use_tidpo = getattr(self.config.loss, 'use_tidpo', False)
        
        if use_tidpo:
            # Calculate token importance weights
            weight_matrix = self._compute_token_importance_weights(
                model, concatenated_batch['concatenated_input_ids'], 
                concatenated_batch['concatenated_attention_mask']
            )
            all_logps_margin, all_position_kl, all_logps = _weighted_tdpo_get_batch_logps(
                all_logits, reference_all_logits, concatenated_batch['concatenated_labels'], 
                weight_matrix, average_log_prob=False
            )
        else:
            # Original TDPO calculation
            all_logps_margin, all_position_kl, all_logps = _tdpo_get_batch_logps(
                all_logits, reference_all_logits, concatenated_batch['concatenated_labels'], 
                average_log_prob=False
            )

        chosen_logps_margin = all_logps_margin[:batch['chosen_input_ids'].shape[0]]
        rejected_logps_margin = all_logps_margin[batch['chosen_input_ids'].shape[0]:]
        chosen_position_kl = all_position_kl[:batch['chosen_input_ids'].shape[0]]
        rejected_position_kl = all_position_kl[batch['chosen_input_ids'].shape[0]:]

        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]].detach()
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:].detach()
        
        # Calculate triplet loss (if TIDPO is enabled)
        triplet_loss = None
        if use_tidpo:
            alpha_triplet = getattr(self.config.loss, 'alpha_triplet', 0.1)
            if alpha_triplet > 0:  # Only calculate triplet loss when alpha_triplet > 0
                bsz = batch['chosen_input_ids'].shape[0]
                triplet_loss = self._compute_triplet_loss(
                    model,
                    reference_model,
                    batch,
                    chosen_logits=all_logits[:bsz],
                    rejected_logits=all_logits[bsz:],
                    chosen_ref_logits=reference_all_logits[:bsz],
                    rejected_ref_logits=reference_all_logits[bsz:],
                )
            else:
                triplet_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        return chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, \
            chosen_logps, rejected_logps, triplet_loss

    def dpo_concatenated_forward(self, model: nn.Module, reference_model: nn.Module,
                                batch: Dict[str, Union[List, torch.LongTensor]]):
        """Forward pass for (weighted) DPO used by TI-DPO.

        Returns:
            chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, chosen_logps, rejected_logps, triplet_loss
        """
        concatenated_batch = concatenated_inputs(batch, pad_token_id=int(self.tokenizer.pad_token_id))

        all_logits = model(concatenated_batch['concatenated_input_ids'],
                           attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)
        with torch.no_grad():
            reference_all_logits = reference_model(concatenated_batch['concatenated_input_ids'],
                                                   attention_mask=concatenated_batch['concatenated_attention_mask']).logits.to(torch.float32)

        kl_coef = float(getattr(self.config.loss, 'kl_coef', 0.0))
        if kl_coef != 0.0:
            all_position_kl = _position_kl_sum(
                all_logits,
                reference_all_logits,
                concatenated_batch['concatenated_labels'],
            )
        else:
            all_position_kl = torch.zeros(all_logits.shape[0], device=all_logits.device, dtype=all_logits.dtype)

        use_tidpo = getattr(self.config.loss, 'use_tidpo', True)
        if use_tidpo:
            weight_matrix = self._compute_token_importance_weights(
                model,
                concatenated_batch['concatenated_input_ids'],
                concatenated_batch['concatenated_attention_mask'],
                labels=concatenated_batch['concatenated_labels'],
            )
            all_logps_margin, all_logps = _weighted_dpo_get_batch_logps(
                all_logits,
                reference_all_logits,
                concatenated_batch['concatenated_labels'],
                weight_matrix,
                average_log_prob=False,
            )
        else:
            all_logps_margin, all_logps = _dpo_get_batch_logps(
                all_logits,
                reference_all_logits,
                concatenated_batch['concatenated_labels'],
                average_log_prob=False,
            )

        bsz = batch['chosen_input_ids'].shape[0]
        chosen_logps_margin = all_logps_margin[:bsz]
        rejected_logps_margin = all_logps_margin[bsz:]
        chosen_position_kl = all_position_kl[:bsz]
        rejected_position_kl = all_position_kl[bsz:]
        chosen_logps = all_logps[:bsz].detach()
        rejected_logps = all_logps[bsz:].detach()

        triplet_loss = None
        alpha_triplet = getattr(self.config.loss, 'alpha_triplet', 0.0)
        if alpha_triplet and alpha_triplet > 0:
            triplet_loss = self._compute_triplet_loss(
                model,
                reference_model,
                batch,
                chosen_logits=all_logits[:bsz],
                rejected_logits=all_logits[bsz:],
                chosen_ref_logits=reference_all_logits[:bsz],
                rejected_ref_logits=reference_all_logits[bsz:],
            )
        else:
            triplet_loss = torch.tensor(0.0, device=next(model.parameters()).device)

        return chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, chosen_logps, rejected_logps, triplet_loss

    def _compute_triplet_loss(self, model: nn.Module, reference_model: nn.Module,
                             batch: Dict[str, Union[List, torch.LongTensor]],
                             chosen_logits: Optional[torch.Tensor] = None,
                             rejected_logits: Optional[torch.Tensor] = None,
                             chosen_ref_logits: Optional[torch.Tensor] = None,
                             rejected_ref_logits: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute triplet loss (Equation 14)"""
        try:
            # 1. Generate intermediate outputs (anchor)
            anchor_outputs = self._generate_anchor_outputs(model, batch)

            # 2. Build response-only masks.
            # For chosen/rejected we can rely on labels to isolate response tokens.
            # For anchor, we mask out the prompt region and keep only generated tokens.
            pad_id = int(self.tokenizer.pad_token_id)

            pos_mask_full = (batch['chosen_labels'] != -100) if 'chosen_labels' in batch else (batch['chosen_input_ids'] != pad_id)
            neg_mask_full = (batch['rejected_labels'] != -100) if 'rejected_labels' in batch else (batch['rejected_input_ids'] != pad_id)

            if 'prompt_input_ids' in batch and 'prompt_attention_mask' in batch:
                prompt_seq_len = int(batch['prompt_input_ids'].shape[1])
                anchor_mask_full = torch.zeros_like(anchor_outputs, dtype=torch.bool)
                if anchor_outputs.shape[1] > prompt_seq_len:
                    anchor_mask_full[:, prompt_seq_len:] = (anchor_outputs[:, prompt_seq_len:] != pad_id)
            else:
                anchor_mask_full = (anchor_outputs != pad_id)

            # 3. Calculate log-ratio sequences (response-only)
            # Anchor always needs an extra forward.
            d_anchor, m_anchor = self._get_log_ratio_sequence(
                model,
                reference_model,
                anchor_outputs,
                token_mask=anchor_mask_full,
                policy_requires_grad=True,
            )

            # For chosen/rejected, reuse concatenated-forward logits if provided.
            if chosen_logits is not None and chosen_ref_logits is not None:
                d_pos, m_pos = self._log_ratio_sequence_from_logits(
                    logits=chosen_logits,
                    ref_logits=chosen_ref_logits,
                    input_ids=batch['chosen_input_ids'],
                    token_mask=pos_mask_full,
                )
            else:
                d_pos, m_pos = self._get_log_ratio_sequence(
                    model,
                    reference_model,
                    batch['chosen_input_ids'],
                    token_mask=pos_mask_full,
                    policy_requires_grad=True,
                )

            if rejected_logits is not None and rejected_ref_logits is not None:
                d_neg, m_neg = self._log_ratio_sequence_from_logits(
                    logits=rejected_logits,
                    ref_logits=rejected_ref_logits,
                    input_ids=batch['rejected_input_ids'],
                    token_mask=neg_mask_full,
                )
            else:
                d_neg, m_neg = self._get_log_ratio_sequence(
                    model,
                    reference_model,
                    batch['rejected_input_ids'],
                    token_mask=neg_mask_full,
                    policy_requires_grad=True,
                )
            
            # 3. Check if tensors are empty
            if d_anchor.numel() == 0 or d_pos.numel() == 0 or d_neg.numel() == 0:
                print("Warning: One or more log-ratio sequences are empty, skipping triplet loss calculation")
                return torch.tensor(0.0, device=next(model.parameters()).device)
            
            # 4. Ensure all tensors have the same sequence length (packed sequences are right-padded)
            max_len = max(d_anchor.shape[1], d_pos.shape[1], d_neg.shape[1])

            def _pad_2d(x: torch.Tensor, target_len: int, pad_value: float = 0.0) -> torch.Tensor:
                if x.shape[1] >= target_len:
                    return x
                pad = torch.full((x.shape[0], target_len - x.shape[1]), pad_value, device=x.device, dtype=x.dtype)
                return torch.cat([x, pad], dim=1)

            d_anchor = _pad_2d(d_anchor, max_len, 0.0)
            d_pos = _pad_2d(d_pos, max_len, 0.0)
            d_neg = _pad_2d(d_neg, max_len, 0.0)
            m_anchor = _pad_2d(m_anchor.to(torch.float32), max_len, 0.0).to(torch.bool)
            m_pos = _pad_2d(m_pos.to(torch.float32), max_len, 0.0).to(torch.bool)
            m_neg = _pad_2d(m_neg.to(torch.float32), max_len, 0.0).to(torch.bool)
            
            # 5. Check numerical stability
            if torch.isnan(d_anchor).any() or torch.isnan(d_pos).any() or torch.isnan(d_neg).any():
                print("Warning: Log-ratio sequences contain NaN values, skipping triplet loss calculation")
                return torch.tensor(0.0, device=next(model.parameters()).device)
            
            # 6. Calculate triplet loss (mask-aware)
            # ||d_anchor - d_pos||² - ||d_anchor - d_neg||² + α_trp
            diff_pos = d_anchor - d_pos
            diff_neg = d_anchor - d_neg

            mask_pos = (m_anchor & m_pos).to(diff_pos.dtype)
            mask_neg = (m_anchor & m_neg).to(diff_neg.dtype)

            dist_pos = torch.sum((diff_pos ** 2) * mask_pos, dim=-1)  # [B]
            dist_neg = torch.sum((diff_neg ** 2) * mask_neg, dim=-1)  # [B]
            
            # Check numerical stability of distance calculations
            if torch.isnan(dist_pos).any() or torch.isnan(dist_neg).any():
                print("Warning: Distance calculations contain NaN values, skipping triplet loss calculation")
                return torch.tensor(0.0, device=next(model.parameters()).device)
            
            # Apply hinge loss and margin
            alpha_triplet = getattr(self.config.loss, 'alpha_triplet', 0.1)
            triplet_loss = F.relu(dist_pos - dist_neg + alpha_triplet).mean()
            
            # Check final loss value
            if torch.isnan(triplet_loss) or torch.isinf(triplet_loss):
                print("Warning: Triplet loss is NaN or Inf, returning zero loss")
                return torch.tensor(0.0, device=next(model.parameters()).device)
            
            return triplet_loss
            
        except Exception as e:
            print(f"Triplet loss calculation failed: {e}")
            # Return zero loss as fallback
            return torch.tensor(0.0, device=next(model.parameters()).device)

    def _generate_anchor_outputs(self, model: nn.Module, 
                                batch: Dict[str, Union[List, torch.LongTensor]]) -> torch.LongTensor:
        """Generate intermediate outputs (anchor) for triplet loss calculation"""
        try:
            anchor_top_k = int(getattr(self.config.loss, 'anchor_top_k', 50))
            anchor_top_p = float(getattr(self.config.loss, 'anchor_top_p', 0.95))
            anchor_temperature = float(getattr(self.config.loss, 'anchor_temperature', 0.8))

            # Prefer sampling an anchor response from the same prompt.
            # Using chosen as the prompt often degenerates (chosen is already max_length padded),
            # which makes anchor==chosen and collapses the triplet objective.

            if 'prompt_input_ids' in batch and 'prompt_attention_mask' in batch:
                prompt_input_ids, prompt_attention_mask = self._left_pad_for_generation(
                    batch['prompt_input_ids'],
                    batch['prompt_attention_mask'],
                )

                max_prompt_len = int(prompt_attention_mask.to(torch.long).sum(dim=1).max().item())
                remaining = max(1, int(self.config.max_length) - max_prompt_len)
                max_new_tokens_cap = int(getattr(self.config.loss, 'anchor_max_new_tokens', 64))
                max_new_tokens = min(max_new_tokens_cap, remaining)

                # FSDP generation guard
                ctx = lambda: (FSDP.summon_full_params(model, writeback=False,
                                                       recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
                with ctx(), torch.no_grad():
                    anchor_outputs = model.generate(
                        prompt_input_ids,
                        attention_mask=prompt_attention_mask,
                        do_sample=True,
                        top_k=anchor_top_k,
                        top_p=anchor_top_p,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        temperature=anchor_temperature
                    )

                anchor_outputs = pad_to_length(anchor_outputs, self.config.max_length, self.tokenizer.pad_token_id)
                return anchor_outputs

            # Fallback: if prompts are not available, sample as a continuation of chosen, but
            # compute max_new_tokens from the *actual* (un-padded) chosen length.
            chosen_input_ids, chosen_attention_mask = self._left_pad_for_generation(
                batch['chosen_input_ids'],
                batch['chosen_attention_mask'],
            )

            max_chosen_len = int(chosen_attention_mask.to(torch.long).sum(dim=1).max().item())
            remaining = max(1, int(self.config.max_length) - max_chosen_len)
            max_new_tokens_cap = int(getattr(self.config.loss, 'anchor_max_new_tokens', 64))
            max_new_tokens = min(max_new_tokens_cap, remaining)

            ctx = lambda: (FSDP.summon_full_params(model, writeback=False,
                                                   recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx(), torch.no_grad():
                anchor_outputs = model.generate(
                    chosen_input_ids,
                    attention_mask=chosen_attention_mask,
                    do_sample=True,
                    top_k=anchor_top_k,
                    top_p=anchor_top_p,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    temperature=anchor_temperature
                )

            anchor_outputs = pad_to_length(anchor_outputs, self.config.max_length, self.tokenizer.pad_token_id)
            return anchor_outputs
        except Exception as e:
            print(f"Anchor generation failed: {e}")
            # If generation fails, use chosen_input_ids as fallback
            return batch['chosen_input_ids']

    def _get_log_ratio_sequence(self, model: nn.Module, reference_model: nn.Module, 
                               input_ids: torch.LongTensor,
                               token_mask: Optional[torch.Tensor] = None,
                               policy_requires_grad: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate log-ratio sequence d = [log(π_θ/π_ref) for each token]"""
        try:
            # Validate input
            if input_ids.numel() == 0:
                z = torch.zeros(0, 0, device=input_ids.device)
                m = torch.zeros(0, 0, device=input_ids.device, dtype=torch.bool)
                return z, m
            
            batch_size, seq_len = input_ids.shape
            if seq_len <= 1:
                z = torch.zeros(batch_size, 0, device=input_ids.device)
                m = torch.zeros(batch_size, 0, device=input_ids.device, dtype=torch.bool)
                return z, m
            
            # Ensure input_ids are within valid range
            vocab_size = model.config.vocab_size
            if input_ids.max() >= vocab_size or input_ids.min() < 0:
                print(f"Warning: input_ids contains invalid token IDs, range: [{input_ids.min()}, {input_ids.max()}], vocab_size: {vocab_size}")
                # Replace invalid tokens with pad_token_id
                input_ids = torch.where(
                    (input_ids >= 0) & (input_ids < vocab_size),
                    input_ids,
                    torch.tensor(self.tokenizer.pad_token_id, device=input_ids.device, dtype=input_ids.dtype)
                )
            
            # Attention mask for model forward (pad-based)
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

            # Token mask for selecting positions in the sequence (response-only when provided)
            if token_mask is None:
                token_mask = attention_mask.to(torch.bool)
            else:
                if token_mask.shape != input_ids.shape:
                    raise ValueError(f"token_mask shape mismatch: token_mask={token_mask.shape}, input_ids={input_ids.shape}")
                token_mask = token_mask.to(torch.bool)
            
            # Forward pass
            # IMPORTANT: triplet loss must backprop through policy logits, so we must not
            # wrap the policy forward in torch.no_grad(). Reference forward can stay no_grad.
            if policy_requires_grad:
                logits = model(input_ids, attention_mask=attention_mask).logits
            else:
                with torch.no_grad():
                    logits = model(input_ids, attention_mask=attention_mask).logits

            with torch.no_grad():
                ref_logits = reference_model(input_ids, attention_mask=attention_mask).logits
            
            # Calculate log probabilities
            logp = F.log_softmax(logits, dim=-1)  # [B, L, V]
            ref_logp = F.log_softmax(ref_logits, dim=-1)  # [B, L, V]
            
            # Get log probabilities of actual tokens
            # Skip first token (usually BOS), use subsequent tokens
            seq_ids = input_ids[..., 1:]  # [B, L-1]
            
            # Ensure seq_ids are within valid range
            seq_ids = torch.clamp(seq_ids, 0, vocab_size - 1)
            
            logp_t = torch.gather(logp[..., :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
            refp_t = torch.gather(ref_logp[..., :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
            
            # Calculate log-ratio
            log_ratio = logp_t - refp_t  # [B, L-1]
            
            # Apply response-only mask (shifted)
            mask = token_mask[..., 1:]  # [B, L-1]
            log_ratio = log_ratio * mask.to(log_ratio.dtype)

            # Pack masked positions so distances are computed over response-token indices
            # (not absolute prompt+response positions).
            packed_list: List[torch.Tensor] = []
            packed_mask_list: List[torch.Tensor] = []
            max_kept = 0
            for i in range(batch_size):
                kept = log_ratio[i][mask[i]]
                packed_list.append(kept)
                max_kept = max(max_kept, int(kept.numel()))

            if max_kept == 0:
                z = torch.zeros(batch_size, 0, device=input_ids.device, dtype=log_ratio.dtype)
                m = torch.zeros(batch_size, 0, device=input_ids.device, dtype=torch.bool)
                return z, m

            packed_batch = []
            packed_mask_batch = []
            for i in range(batch_size):
                kept = packed_list[i]
                k = int(kept.numel())
                if k < max_kept:
                    pad = torch.zeros(max_kept - k, device=kept.device, dtype=kept.dtype)
                    out = torch.cat([kept, pad], dim=0)
                    m = torch.cat([
                        torch.ones(k, device=kept.device, dtype=torch.bool),
                        torch.zeros(max_kept - k, device=kept.device, dtype=torch.bool),
                    ], dim=0)
                else:
                    out = kept
                    m = torch.ones(max_kept, device=kept.device, dtype=torch.bool)
                packed_batch.append(out)
                packed_mask_batch.append(m)

            return torch.stack(packed_batch, dim=0), torch.stack(packed_mask_batch, dim=0)
            
        except Exception as e:
            print(f"Log-ratio calculation failed: {e}")
            # Return zero tensor as fallback
            batch_size, seq_len = input_ids.shape
            z = torch.zeros(batch_size, max(0, seq_len-1), device=input_ids.device)
            m = torch.zeros(batch_size, max(0, seq_len-1), device=input_ids.device, dtype=torch.bool)
            return z, m

    def _log_ratio_sequence_from_logits(self,
                                       logits: torch.Tensor,
                                       ref_logits: torch.Tensor,
                                       input_ids: torch.LongTensor,
                                       token_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute packed log-ratio sequence from precomputed logits.

        This avoids extra forward passes for chosen/rejected in triplet loss.
        """
        # Handle empty/short sequences
        if input_ids.numel() == 0:
            z = torch.zeros(0, 0, device=input_ids.device)
            m = torch.zeros(0, 0, device=input_ids.device, dtype=torch.bool)
            return z, m

        batch_size, seq_len = input_ids.shape
        if seq_len <= 1:
            z = torch.zeros(batch_size, 0, device=input_ids.device)
            m = torch.zeros(batch_size, 0, device=input_ids.device, dtype=torch.bool)
            return z, m

        if token_mask.shape != input_ids.shape:
            raise ValueError(f"token_mask shape mismatch: token_mask={token_mask.shape}, input_ids={input_ids.shape}")

        # logits/ref_logits are [B, L, V]
        vocab_size = logits.shape[-1]
        seq_ids = input_ids[..., 1:].clamp(0, vocab_size - 1)  # [B, L-1]

        logp = F.log_softmax(logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        logp_t = torch.gather(logp[..., :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        refp_t = torch.gather(ref_logp[..., :-1, :], 2, seq_ids.unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        log_ratio = logp_t - refp_t

        mask = token_mask.to(torch.bool)[..., 1:]  # [B, L-1]
        log_ratio = log_ratio * mask.to(log_ratio.dtype)

        # Pack masked positions so distances are computed over response-token indices
        packed_list: List[torch.Tensor] = []
        max_kept = 0
        for i in range(batch_size):
            kept = log_ratio[i][mask[i]]
            packed_list.append(kept)
            max_kept = max(max_kept, int(kept.numel()))

        if max_kept == 0:
            z = torch.zeros(batch_size, 0, device=input_ids.device, dtype=log_ratio.dtype)
            m = torch.zeros(batch_size, 0, device=input_ids.device, dtype=torch.bool)
            return z, m

        packed_batch = []
        packed_mask_batch = []
        for i in range(batch_size):
            kept = packed_list[i]
            k = int(kept.numel())
            if k < max_kept:
                pad = torch.zeros(max_kept - k, device=kept.device, dtype=kept.dtype)
                out = torch.cat([kept, pad], dim=0)
                m = torch.cat([
                    torch.ones(k, device=kept.device, dtype=torch.bool),
                    torch.zeros(max_kept - k, device=kept.device, dtype=torch.bool),
                ], dim=0)
            else:
                out = kept
                m = torch.ones(max_kept, device=kept.device, dtype=torch.bool)
            packed_batch.append(out)
            packed_mask_batch.append(m)

        return torch.stack(packed_batch, dim=0), torch.stack(packed_mask_batch, dim=0)

    def _compute_token_importance_weights(self, model: nn.Module, input_ids: torch.LongTensor, 
                                         attention_mask: torch.LongTensor,
                                         labels: Optional[torch.LongTensor] = None) -> torch.FloatTensor:
        """Compute the importance weight matrix for each token"""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize weight matrix
        weight_matrix = torch.ones(batch_size, seq_len, device=device)

        try:
            # Compute gradient attribution directly on token IDs (avoids decode/re-tokenize mismatch).
            enable_attr = getattr(self.config.loss, 'enable_gradient_attribution', True)
            lambda_importance = float(getattr(self.config.loss, 'lambda_importance', 0.8))
            lambda_importance = float(max(0.0, min(1.0, lambda_importance)))
            prior_sigma_div = float(getattr(self.config.loss, 'prior_sigma_div', 4.0))
            prior_sigma_div = float(max(1.0, prior_sigma_div))

            if enable_attr:
                # NOTE: evaluation loops may wrap forward passes in `torch.no_grad()`.
                # Gradient attribution requires gradients, so explicitly re-enable them here.
                with torch.enable_grad():
                    importances = gradient_attribution.compute_language_model_gradient_attribution_from_ids(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        device=device,
                    )
            else:
                importances = torch.zeros(batch_size, seq_len, device=device, dtype=torch.float32)

            for i in range(batch_size):
                valid_mask = attention_mask[i].to(torch.bool)
                if labels is not None:
                    valid_mask = valid_mask & (labels[i] != -100)

                valid_idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
                if valid_idx.numel() <= 1:
                    weight_matrix[i].zero_()
                    continue

                scores = importances[i][valid_idx].to(torch.float32).clamp_min(0)
                if scores.sum() > 0:
                    norm_scores = scores / scores.sum()
                else:
                    norm_scores = None

                # Gaussian prior over the *valid* token span
                n = int(valid_idx.numel())
                pos = torch.arange(n, device=device, dtype=torch.float32)
                center = (n - 1) / 2.0
                sigma = max(1.0, n / prior_sigma_div)
                prior = torch.exp(-0.5 * ((pos - center) / sigma) ** 2)
                prior = prior / prior.sum()

                if norm_scores is not None:
                    mixed = lambda_importance * norm_scores + (1.0 - lambda_importance) * prior
                else:
                    mixed = prior
                mixed = mixed / mixed.sum()

                # IMPORTANT: for training stability and to keep the scale of the DPO margin comparable
                # to the unweighted sum, we want weights to have mean 1 over the valid token span
                # (i.e. sum = number of valid tokens), not sum to 1.
                mixed = mixed * float(n)

                weight_matrix[i].zero_()
                weight_matrix[i][valid_idx] = mixed

        except Exception as e:
            print(f"Token importance weight calculation failed: {e}")
            weight_matrix = torch.ones(batch_size, seq_len, device=device)

        return weight_matrix

    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute the SFT or TDPO loss and other metrics for the given batch of inputs."""

        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name == 'tdpo':
            forward_output = self.tdpo_concatenated_forward(self.policy, self.reference_model, batch)
            if len(forward_output) == 7:
                chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps, triplet_loss = forward_output
            else:
                chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps = forward_output
                triplet_loss = None

            dpo_losses, chosen_rewards, rejected_rewards = tdpo_loss(
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                beta=loss_config.beta,
                alpha=loss_config.alpha,
                if_tdpo2=loss_config.if_tdpo2,
            )

            final_losses = dpo_losses
            metrics[f'triplet_loss_{train_test}'] = 0.0

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
            all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)

            metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
            metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
            losses = final_losses

        elif loss_config.name == 'tidpo':
            # TI-DPO builds on the TDPO-style objective (same base as the repo's TDPO baseline),
            # but uses token-importance weighting (controlled by `use_tidpo`) and optionally adds
            # the triplet term.
            chosen_logps_margin, rejected_logps_margin, chosen_position_kl, rejected_position_kl, policy_chosen_logps, policy_rejected_logps, triplet_loss = self.tdpo_concatenated_forward(
                self.policy,
                self.reference_model,
                batch,
            )

            base_losses, chosen_rewards, rejected_rewards = tdpo_loss(
                chosen_logps_margin,
                rejected_logps_margin,
                chosen_position_kl,
                rejected_position_kl,
                beta=loss_config.beta,
                alpha=float(getattr(loss_config, 'alpha', 0.5)),
                if_tdpo2=bool(getattr(loss_config, 'if_tdpo2', True)),
            )

            gamma = getattr(loss_config, 'gamma', 0.1)
            final_losses = base_losses + gamma * triplet_loss

            # Debug-only: prove triplet contributes gradients to policy.
            if getattr(self.config, 'debug', False):
                try:
                    metrics[f'debug_{train_test}/triplet_requires_grad'] = int(bool(getattr(triplet_loss, 'requires_grad', False)))

                    probe_param = None
                    for p in self.policy.parameters():
                        if p.requires_grad and p.dtype.is_floating_point and p.numel() > 0:
                            probe_param = p
                            break

                    if probe_param is None:
                        metrics[f'debug_{train_test}/triplet_grad_probe_found_param'] = 0
                        metrics[f'debug_{train_test}/triplet_grad_probe_nonzero'] = 0
                    else:
                        metrics[f'debug_{train_test}/triplet_grad_probe_found_param'] = 1
                        g = torch.autograd.grad(
                            outputs=triplet_loss,
                            inputs=probe_param,
                            retain_graph=True,
                            create_graph=False,
                            allow_unused=True,
                        )[0]
                        if g is None:
                            metrics[f'debug_{train_test}/triplet_grad_probe_nonzero'] = 0
                        else:
                            gn = float(g.detach().float().norm().cpu().item())
                            metrics[f'debug_{train_test}/triplet_grad_probe_norm'] = gn
                            metrics[f'debug_{train_test}/triplet_grad_probe_nonzero'] = int(gn > 0.0)
                except Exception as e:
                    metrics[f'debug_{train_test}/triplet_grad_probe_error'] = str(e)

            # Optional KL regularization (keeps DPO-style objective, but adds a controllable KL penalty)
            kl_coef = float(getattr(loss_config, 'kl_coef', 0.0))
            if kl_coef != 0.0:
                kl_term = 0.5 * (chosen_position_kl + rejected_position_kl)
                final_losses = final_losses + kl_coef * kl_term
                all_device_chosen_position_kl = all_gather_if_needed(chosen_position_kl.detach(), self.rank, self.world_size)
                all_device_rejected_position_kl = all_gather_if_needed(rejected_position_kl.detach(), self.rank, self.world_size)
                metrics[f'kl_{train_test}/chosen'] = all_device_chosen_position_kl.cpu().numpy().tolist()
                metrics[f'kl_{train_test}/rejected'] = all_device_rejected_position_kl.cpu().numpy().tolist()
                metrics[f'kl_{train_test}/margin'] = (all_device_chosen_position_kl - all_device_rejected_position_kl).cpu().numpy().tolist()
            metrics[f'kl_coef_{train_test}'] = kl_coef
            metrics[f'triplet_loss_{train_test}'] = triplet_loss.detach().cpu().numpy().tolist()
            metrics[f'gamma_{train_test}'] = gamma

            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
            losses = final_losses

        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'],
                                               attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)

            losses = -policy_chosen_logps

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()

        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def train(self):
        """Begin either SFT or TDPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy.parameters(), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0,
                                                                                                      (step + 1) / (
                                                                                                              self.config.warmup_steps + 1)))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in ('tdpo', 'tidpo'):
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter % self.config.eval_every == 0 and (
                    self.example_counter > 0 or self.config.do_first_eval):
                rank0_print(f'Running evaluation after {self.example_counter} train examples')
                self.policy.eval()

                all_eval_metrics = defaultdict(list)
                if self.config.sample_during_eval:
                    all_policy_samples, all_reference_samples = [], []
                    policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                    if self.config.loss.name in ('tdpo', 'tidpo'):
                        reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

                for eval_batch in (
                tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
                    local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size,
                                                                       self.rank)
                    with torch.no_grad():
                        _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

                    for k, v in eval_metrics.items():
                        if isinstance(v, list):
                            all_eval_metrics[k].extend(v)
                        else:
                            all_eval_metrics[k].append(v)

                if self.config.sample_during_eval:
                    if self.config.n_eval_model_samples < self.config.eval_batch_size:
                        rank0_print(
                            f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                        sample_batches = self.eval_batches[:1]
                    else:
                        n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                        sample_batches = self.eval_batches[:n_sample_batches]
                    for eval_batch in (
                    tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                        local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size,
                                                                           self.rank)
                        policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                        all_policy_samples.extend(policy_samples)
                        all_reference_samples.extend(reference_samples)

                        for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                            policy_text_table.add_data(self.example_counter, prompt, sample)
                        if self.config.loss.name in ('tdpo', 'tidpo'):
                            for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                                reference_text_table.add_data(self.example_counter, prompt, sample)

                mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
                rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
                if self.config.sample_during_eval:
                    rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                    if self.config.loss.name in ('tdpo', 'tidpo'):
                        rank0_print(json.dumps(all_reference_samples[:10], indent=2))

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_eval_metrics, step=self.example_counter)

                    if self.config.sample_during_eval:
                        wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                        if self.config.loss.name in ('tdpo', 'tidpo'):
                            wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)
                
                print(f"Example Counter: {self.example_counter} \nSave Dir: {self.run_dir}")
                if self.example_counter > 0:
                    if self.config.debug:
                        rank0_print('skipping save in debug mode')
                    else:
                        output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                        rank0_print(f'creating checkpoint to write to {output_dir}...')
                        self.save(output_dir, mean_eval_metrics)
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                try:
                    global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx,
                                                                        self.config.gradient_accumulation_steps, self.rank)
                    local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size,
                                                                       self.rank)
                    loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                    (loss / self.config.gradient_accumulation_steps).backward()

                    for k, v in metrics.items():
                        if isinstance(v, list):
                            batch_metrics[k].extend(v)
                        else:
                            batch_metrics[k].append(v)
                except ValueError as e:
                    if "Empty batch" in str(e) or "Invalid slice" in str(e):
                        print(f"Skipping empty microbatch {microbatch_idx}: {e}")
                        continue
                    else:
                        raise e

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str,
                         dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""
        if output_dir is None:
            output_dir = os.path.join(self.run_dir, 'LATEST')
        os.makedirs(output_dir, exist_ok=True)

        # # 1. SAVE THE ENTIRE MODEL
        # full_model_path = os.path.join(output_dir, 'full_policy.pt')
        # rank0_print(f'writing entire model to {full_model_path}...')
        # torch.save(self.policy, full_model_path)

        # 2. SAVE OPTIMIZER & SCHEDULER STATES (Keep as state_dicts)
        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)

class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str,
                 reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.

           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class}, )

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper,
                                               check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in ('tdpo', 'tidpo'):
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)

        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in ('tdpo', 'tidpo'):
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()

        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
