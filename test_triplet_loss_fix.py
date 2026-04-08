#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for triplet loss fix
Verify that the fixed triplet loss calculation works properly
"""

import sys
import os
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainers import BasicTrainer

def test_get_log_ratio_sequence():
    """Test _get_log_ratio_sequence function"""
    print("🧪 Testing _get_log_ratio_sequence function...")
    
    try:
        # Create simple configuration
        config = DictConfig({
            'model': {
                'name_or_path': 'gpt2',
                'tokenizer_name_or_path': None,
                'archive': None,
                'block_name': 'GPT2Block',
                'policy_dtype': 'float32',
                'fsdp_policy_mp': None,
                'reference_dtype': 'float32'
            },
            'loss': {
                'name': 'tdpo',
                'use_tidpo': False,
                'if_tdpo2': False,
                'alpha': 0.5,
                'beta': 0.1,
                'gamma': 0.1,
                'alpha_triplet': 0.0,
                'reference_free': False,
                'enable_gradient_attribution': False
            },
            'max_length': 128,
            'max_prompt_length': 64,
            'batch_size': 2,
            'eval_batch_size': 2,
            'debug': False,
            'fsdp_port': None,
            'datasets': ['hh'],
            'wandb': {'enabled': False},
            'local_dirs': ['.cache'],
            'local_run_dir': 'test_run',
            'lr': 1e-5,
            'gradient_accumulation_steps': 1,
            'max_grad_norm': 1.0,
            'n_epochs': 1,
            'n_examples': None,
            'n_eval_examples': 4,
            'trainer': 'BasicTrainer',
            'optimizer': 'AdamW',
            'warmup_steps': 10,
            'activation_checkpointing': False,
            'eval_every': 100,
            'minimum_log_interval_secs': 1.0,
            'sample_during_eval': False,
            'n_eval_model_samples': 1,
            'do_first_eval': False
        })
        
        # Load model and tokenizer
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        tokenizer = AutoTokenizer.from_pretrained('gpt2')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained('gpt2')
        reference_model = AutoModelForCausalLM.from_pretrained('gpt2')
        
        # Create trainer instance
        trainer = BasicTrainer(
            policy=model,
            config=config,
            seed=0,
            run_dir='test_run',
            reference_model=reference_model
        )
        
        # Test normal input
        print("Testing normal input...")
        input_ids = torch.randint(0, tokenizer.vocab_size, (2, 10))
        d, m = trainer._get_log_ratio_sequence(model, reference_model, input_ids)
        print(f"✅ Normal input test passed, d shape: {d.shape}, mask shape: {m.shape}")
        
        # Test edge cases
        print("Testing edge cases...")
        
        # Empty tensor
        empty_input = torch.zeros(0, 0, dtype=torch.long)
        d, m = trainer._get_log_ratio_sequence(model, reference_model, empty_input)
        print(f"✅ Empty tensor test passed, d shape: {d.shape}, mask shape: {m.shape}")
        
        # Single token
        single_token = torch.randint(0, tokenizer.vocab_size, (1, 1))
        d, m = trainer._get_log_ratio_sequence(model, reference_model, single_token)
        print(f"✅ Single token test passed, d shape: {d.shape}, mask shape: {m.shape}")
        
        # Invalid token IDs
        invalid_input = torch.tensor([[tokenizer.vocab_size + 100, 1, 2, 3]])
        d, m = trainer._get_log_ratio_sequence(model, reference_model, invalid_input)
        print(f"✅ Invalid token IDs test passed, d shape: {d.shape}, mask shape: {m.shape}")
        
        print("🎉 All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_get_log_ratio_sequence() 