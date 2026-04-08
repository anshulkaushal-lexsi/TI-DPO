#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Gradient Attribution for Language Models

Calculate the importance score of each token in the input text through gradient attribution (Gradient-Based Attribution).
This script uses Hugging Face Transformers and PyTorch to perform gradient sensitivity attribution on any pre-trained classification model that supports inputs_embeds,
outputting the influence intensity of each token on the model's final prediction (or specified label).

Dependencies:
- torch
- transformers
- numpy

Usage example:
python gradient_attribution.py --model_name_or_path gpt2 --text "Hello world"

Parameter description:
--model_name_or_path: Pre-trained model name on Hugging Face or local path (must support inputs_embeds).
--text: Input text to be attributed (single sentence or multiple sentences).
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def compute_language_model_gradient_attribution_from_ids(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.LongTensor | None = None,
    device: torch.device | None = None,
) -> torch.FloatTensor:
    """Compute token importances for a causal LM from token ids.

    This matches the TI-DPO paper's definition:
    - Target: max(logits) at the last *valid* position
    - Gradient: w.r.t. token embeddings
    - Importance: L1 norm of the gradient per token

    Args:
        model: Causal language model (e.g., AutoModelForCausalLM).
        input_ids: (B, L)
        attention_mask: Optional (B, L). If None, inferred by input_ids != pad_token_id when available.
        device: Optional device. If None, inferred from model / input_ids.

    Returns:
        importances: (B, L) float tensor. Pad positions are 0.
    """
    if device is None:
        device = input_ids.device

    input_ids = input_ids.to(device)

    # Infer attention mask if not provided
    if attention_mask is None:
        pad_token_id = getattr(model.config, 'pad_token_id', None)
        if pad_token_id is None:
            # Fallback: treat all positions as valid
            attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)
        else:
            attention_mask = (input_ids != int(pad_token_id)).to(torch.long)
    else:
        attention_mask = attention_mask.to(device)

    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError(f"Expected 2D tensors, got input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")
    if input_ids.shape != attention_mask.shape:
        raise ValueError(f"Shape mismatch: input_ids={input_ids.shape}, attention_mask={attention_mask.shape}")

    original_training = model.training
    model.eval()

    try:
        # Detach to ensure embeddings are leaf and we only take grads w.r.t embeddings.
        embeddings = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)

        outputs = model(inputs_embeds=embeddings, attention_mask=attention_mask)
        logits = outputs.logits  # (B, L, V)

        lengths = attention_mask.to(torch.long).sum(dim=1)  # (B,)
        # Avoid negative indices for empty sequences
        last_pos = torch.clamp(lengths - 1, min=0)

        batch_idx = torch.arange(input_ids.shape[0], device=device)
        last_logits = logits[batch_idx, last_pos, :]  # (B, V)
        target = last_logits.max(dim=-1).values  # (B,)

        grads = torch.autograd.grad(
            outputs=target.sum(),
            inputs=embeddings,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        importances = grads.abs().sum(dim=-1)  # (B, L)
        importances = importances * attention_mask.to(importances.dtype)
        return importances
    finally:
        model.train(original_training)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate the importance score of each token in the text through gradient attribution."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        description="Calculate the importance score of each token in the text through gradient attribution."
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input text to be attributed (single sentence or multiple sentences).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="classification",
        help="Task type, currently only supports 'classification'.",
    )
    parser.add_argument(
        "--label",
        type=int,
        default=None,
        help="(Optional) If task is classification, pass the target label index in integer form; "
        "if not specified, use the model's predicted class for attribution.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="(Optional) Save results to specified file in TSV format: token \\t importance_score.",
    )
    return parser.parse_args()


def compute_gradient_attribution(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    target_label: int = None,
    device: torch.device = torch.device("cpu"),
):
    """
    Perform gradient attribution on input text, calculating the importance score of each token for the final prediction (or specified label).

    Args:
        model: Pre-trained classification model (AutoModelForSequenceClassification).
        tokenizer: Corresponding tokenizer for the model (AutoTokenizer).
        text: Input text to be attributed.
        target_label: (Optional) If specified, perform gradient attribution on the loss/logit of this label;
                     if None, use the model's predicted class index.
        device: torch.device, default CPU; if GPU is available, can pass torch.device("cuda").

    Returns:
        tokens (List[str]): Tokenized tokens (including special tokens like [CLS], [SEP]).
        importances (List[float]): Importance score for each token (using gradient L2 norm).
    """
    model.to(device)
    model.eval()

    # 1. Tokenize input text
    encoding = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        truncation=True,
        max_length=512,
    )
    input_ids = encoding["input_ids"].to(device)           # shape: (1, seq_len)
    attention_mask = encoding["attention_mask"].to(device) # shape: (1, seq_len)

    seq_len = input_ids.size(1)

    # 2. Get embedding layer output (batch_size=1, seq_len, hidden_size)
    #    and make it require gradients
    embeddings = model.get_input_embeddings()(input_ids)  # shape: (1, seq_len, hidden_size)
    embeddings.retain_grad()  # Retain gradients in computation graph
    embeddings.requires_grad_()  # Enable gradient computation for embeddings

    # 3. Forward pass: use inputs_embeds instead of input_ids
    outputs = model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # shape: (1, num_labels)

    # 4. Determine target: if user specified target_label, target that label's logit;
    #    otherwise, use logits.argmax as predicted label
    if target_label is None:
        with torch.no_grad():
            pred_label = torch.argmax(logits, dim=-1).item()
        target_label = pred_label

    # 5. Calculate target_logit (preserve computation graph)
    #    shape: torch.Tensor (scalar)
    target_logit = logits[0, target_label]

    # 6. Backward pass: calculate gradient of logits[target_label] with respect to embeddings
    model.zero_grad()
    # If you want to attribute loss, you can also use cross entropy loss: loss = F.cross_entropy(logits, torch.tensor([target_label]).to(device))
    target_logit.backward(retain_graph=True)

    # 7. Extract gradients: embeddings.grad shape is (1, seq_len, hidden_size)
    grads = embeddings.grad.detach()[0]  # shape: (seq_len, hidden_size)

    # 8. Calculate L2 norm of gradients for each token as importance score
    #    Use torch.norm(grads, p=2, dim=1) => shape: (seq_len,)
    token_importances = torch.norm(grads, p=2, dim=1)  # shape: (seq_len,)

    # 9. Move tensor back to CPU and convert to list
    importances = token_importances.cpu().tolist()

    # 10. Decode input_ids to tokens (including special tokens)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])

    return tokens, importances


def compute_language_model_gradient_attribution(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    text: str,
    device: torch.device = torch.device("cpu"),
):
    """
    Perform gradient attribution on language model, calculating the importance score of each token.
    Specifically designed for causal language models (such as GPT, LLaMA, etc.).

    Args:
        model: Pre-trained language model (AutoModelForCausalLM).
        tokenizer: Corresponding tokenizer for the model (AutoTokenizer).
        text: Input text to be attributed.
        device: torch.device, default CPU; if GPU is available, can pass torch.device("cuda").

    Returns:
        tokens (List[str]): Tokenized tokens.
        importances (List[float]): Importance score for each token (using gradient L1 norm).
    """
    model.to(device)
    
    # Save original model state
    original_training = model.training
    original_requires_grad = {}
    for name, param in model.named_parameters():
        original_requires_grad[name] = param.requires_grad
    
    # Temporarily set to training mode and enable gradient computation
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)

    try:
        # 1. Tokenize input text
        encoding = tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
            truncation=True,
            max_length=512,
        )
        input_ids = encoding["input_ids"].to(device)           # shape: (1, seq_len)
        attention_mask = encoding["attention_mask"].to(device) # shape: (1, seq_len)

        seq_len = input_ids.size(1)

        # 2. Get embedding layer output and make it require gradients
        embeddings = model.get_input_embeddings()(input_ids)  # shape: (1, seq_len, hidden_size)
        
        # Ensure embeddings require gradients
        embeddings.requires_grad_(True)
        embeddings.retain_grad()  # Retain gradients in computation graph

        # 3. Forward pass: use inputs_embeds
        outputs = model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        logits = outputs.logits  # shape: (1, seq_len, vocab_size)

        # 4. For language models, use the last *valid* (non-pad) position as target.
        #    This represents the model's prediction for the next token.
        last_pos = int(attention_mask[0].to(torch.long).sum().item()) - 1
        last_pos = max(last_pos, 0)
        target_logit = logits[0, last_pos, :].max()  # Take the maximum logit at the last valid position

        # 5. Backward pass: calculate gradient of target logit with respect to embeddings
        model.zero_grad()
        target_logit.backward(retain_graph=True)

        # 6. Extract gradients
        if embeddings.grad is not None:
            grads = embeddings.grad.detach()[0]  # shape: (seq_len, hidden_size)
        else:
            # If no gradients, return uniform importance
            grads = torch.ones(seq_len, model.config.hidden_size, device=device)

        # 7. Calculate L1 norm of gradients for each token as importance score
        #    Use L1 norm instead of L2 norm because L1 norm is more sensitive to sparsity
        token_importances = torch.norm(grads, p=1, dim=1)  # shape: (seq_len,)

        # 8. Move tensor back to CPU and convert to list
        importances = token_importances.cpu().tolist()

        # 9. Decode input_ids to tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
    except Exception as e:
        print(f"Gradient attribution calculation failed: {e}")
        # Return uniform importance as fallback
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0]) if 'input_ids' in locals() else []
        importances = [1.0] * len(tokens) if tokens else [1.0]
    
    finally:
        # Restore original model state
        model.train(original_training)
        for name, param in model.named_parameters():
            if name in original_requires_grad:
                param.requires_grad_(original_requires_grad[name])

    return tokens, importances


def main():
    args = parse_args()

    # 1. Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}", file=sys.stderr)

    # 2. Load tokenizer and pre-trained model
    print(f"[Info] Loading model and tokenizer: {args.model_name_or_path}", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path
        )
    except Exception as e:
        print(f"[Error] Cannot load model or tokenizer: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Perform gradient attribution calculation
    print("[Info] Starting gradient attribution calculation...", file=sys.stderr)
    tokens, importances = compute_gradient_attribution(
        model=model,
        tokenizer=tokenizer,
        text=args.text,
        target_label=args.label,
        device=device,
    )

    # 4. Output results (print to stdout)
    print("\n# Token\tImportance (L2 norm of gradient)")
    print("========================================")
    for tok, score in zip(tokens, importances):
        print(f"{tok}\t{score:.6f}")

    # 5. If output_file is specified, write the results in TSV format
    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as fout:
                fout.write("token\timportance\n")
                for tok, score in zip(tokens, importances):
                    fout.write(f"{tok}\t{score:.6f}\n")
            print(f"[Info] Results saved to: {args.output_file}", file=sys.stderr)
        except Exception as e:
            print(f"[Error] Failed to write file: {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
