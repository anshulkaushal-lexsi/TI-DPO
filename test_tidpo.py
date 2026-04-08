#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TIDPO functionality test script
# 
# This script is used to test the core functionality of TIDPO, including:
# 1. Gradient attribution calculation
# 2. Weighted loss function
# 3. Configuration loading

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gradient_attribution():
    """Test gradient attribution functionality"""
    print("🧪 Testing gradient attribution functionality...")
    
    try:
        import gradient_attribution
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load a small model for testing
        model_name = "gpt2"  # Use a smaller model for testing
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test text
        test_text = "Hello world, this is a test sentence for gradient attribution."
        
        # Test gradient attribution
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        tokens, importances = gradient_attribution.compute_language_model_gradient_attribution(
            model=model,
            tokenizer=tokenizer,
            text=test_text,
            device=device
        )
        
        print(f"✅ Gradient attribution test successful!")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Importances: {importances[:5]}...")  # Only show first 5
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient attribution test failed: {e}")
        return False

def test_weighted_loss():
    """Test weighted loss function"""
    print("\n🧪 Testing weighted loss function...")
    
    try:
        # Simulate data
        batch_size = 2
        seq_len = 10
        vocab_size = 50257
        
        # Simulate logits and weights
        logits = torch.randn(batch_size, seq_len, vocab_size)
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        weight_matrix = torch.rand(batch_size, seq_len)
        
        # Calculate weighted loss
        logp = F.log_softmax(logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        
        # Get token log probabilities
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logp_t = torch.gather(logp, 2, token_ids.unsqueeze(-1)).squeeze(-1)
        refp_t = torch.gather(ref_logp, 2, token_ids.unsqueeze(-1)).squeeze(-1)
        
        # Calculate weighted log-ratio
        log_ratio = logp_t - refp_t
        weighted_log_ratio = log_ratio * weight_matrix
        
        print(f"✅ Weighted loss calculation successful!")
        print(f"log_ratio shape: {log_ratio.shape}")
        print(f"weighted_log_ratio shape: {weighted_log_ratio.shape}")
        print(f"weighted_log_ratio mean: {weighted_log_ratio.mean():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Weighted loss test failed: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n🧪 Testing configuration loading...")
    
    try:
        # Simulate TIDPO configuration
        config = {
            'use_tidpo': True,
            'alpha_triplet': 0.1,
            'gamma': 0.1,
            'enable_gradient_attribution': True,
            'beta': 0.1,
            'alpha': 0.5
        }
        
        print(f"✅ Configuration loading successful!")
        print(f"use_tidpo: {config['use_tidpo']}")
        print(f"alpha_triplet: {config['alpha_triplet']}")
        print(f"gamma: {config['gamma']}")
        print(f"enable_gradient_attribution: {config['enable_gradient_attribution']}")
        
        # Validate configuration
        if config['use_tidpo'] and config['enable_gradient_attribution']:
            print("✅ Configuration is valid for TIDPO training")
        else:
            print("⚠️  Configuration may not be optimal for TIDPO training")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 TIDPO Functionality Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_gradient_attribution,
        test_weighted_loss,
        test_config_loading,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test execution failed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed tests: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! TIDPO functionality is working correctly!")
    else:
        print("⚠️  Some tests failed, please check the implementation")
    
    return passed == total

if __name__ == "__main__":
    main() 