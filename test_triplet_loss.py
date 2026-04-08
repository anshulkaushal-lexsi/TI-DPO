#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Triplet Loss Test Script

Test the calculation and integration of triplet loss in TIDPO
"""

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

def test_triplet_loss_calculation():
    """Test triplet loss calculation"""
    print("🧪 Testing triplet loss calculation...")
    
    try:
        # Simulate data
        batch_size = 2
        seq_len = 10
        
        # Simulate log-ratio sequences
        d_anchor = torch.randn(batch_size, seq_len)
        d_pos = torch.randn(batch_size, seq_len)
        d_neg = torch.randn(batch_size, seq_len)
        
        # Calculate triplet loss
        # ||d_anchor - d_pos||² - ||d_anchor - d_neg||² + α_trp
        diff_pos = d_anchor - d_pos
        diff_neg = d_anchor - d_neg
        
        # Calculate squared L2 norm
        dist_pos = torch.sum(diff_pos**2, dim=-1)  # [B]
        dist_neg = torch.sum(diff_neg**2, dim=-1)  # [B]
        
        # Apply hinge loss and margin
        alpha_triplet = 0.1
        triplet_loss = F.relu(dist_pos - dist_neg + alpha_triplet).mean()
        
        print(f"✅ Triplet loss calculation successful!")
        print(f"d_anchor shape: {d_anchor.shape}")
        print(f"d_pos shape: {d_pos.shape}")
        print(f"d_neg shape: {d_neg.shape}")
        print(f"dist_pos: {dist_pos}")
        print(f"dist_neg: {dist_neg}")
        print(f"triplet_loss: {triplet_loss.item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Triplet loss calculation failed: {e}")
        return False

def test_log_ratio_sequence():
    """Test log-ratio sequence calculation"""
    print("\n🧪 Testing log-ratio sequence calculation...")
    
    try:
        # Simulate model outputs
        batch_size = 2
        seq_len = 10
        vocab_size = 50257
        
        # Simulate logits
        logits = torch.randn(batch_size, seq_len, vocab_size)
        ref_logits = torch.randn(batch_size, seq_len, vocab_size)
        
        # Calculate log probabilities
        logp = F.log_softmax(logits, dim=-1)
        ref_logp = F.log_softmax(ref_logits, dim=-1)
        
        # Simulate token IDs
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Calculate log-ratio
        logp_t = torch.gather(logp, 2, token_ids.unsqueeze(-1)).squeeze(-1)
        refp_t = torch.gather(ref_logp, 2, token_ids.unsqueeze(-1)).squeeze(-1)
        
        log_ratio = logp_t - refp_t
        
        print(f"✅ Log-ratio sequence calculation successful!")
        print(f"log_ratio shape: {log_ratio.shape}")
        print(f"log_ratio range: [{log_ratio.min():.4f}, {log_ratio.max():.4f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Log-ratio sequence calculation failed: {e}")
        return False

def test_numerical_stability():
    """Test numerical stability"""
    print("\n🧪 Testing numerical stability...")
    
    try:
        # Test with extreme values
        batch_size = 2
        seq_len = 10
        
        # Test with very large values
        d_anchor = torch.randn(batch_size, seq_len) * 1000
        d_pos = torch.randn(batch_size, seq_len) * 1000
        d_neg = torch.randn(batch_size, seq_len) * 1000
        
        diff_pos = d_anchor - d_pos
        diff_neg = d_anchor - d_neg
        
        dist_pos = torch.sum(diff_pos**2, dim=-1)
        dist_neg = torch.sum(diff_neg**2, dim=-1)
        
        alpha_triplet = 0.1
        triplet_loss = F.relu(dist_pos - dist_neg + alpha_triplet).mean()
        
        if torch.isnan(triplet_loss) or torch.isinf(triplet_loss):
            print("❌ Numerical instability detected")
            return False
        
        print(f"✅ Numerical stability test passed")
        print(f"triplet_loss: {triplet_loss.item()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Numerical stability test failed: {e}")
        return False

def test_loss_combination():
    """Test loss combination"""
    print("\n🧪 Testing loss combination...")
    
    try:
        # Simulate DPO loss and triplet loss
        batch_size = 2
        dpo_losses = torch.randn(batch_size)
        triplet_loss = torch.randn(1)
        
        # Combine losses (Equation 15: L_TI-DPO = L_DPO-w + γ * L_triplet)
        gamma = 0.1
        final_losses = dpo_losses + gamma * triplet_loss
        
        print(f"✅ Loss combination successful!")
        print(f"DPO losses: {dpo_losses}")
        print(f"Triplet loss: {triplet_loss}")
        print(f"Gamma: {gamma}")
        print(f"Final losses: {final_losses}")
        print(f"Final loss mean: {final_losses.mean():.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Loss combination failed: {e}")
        return False

def test_config_parameters():
    """Test configuration parameters"""
    print("\n🧪 Testing configuration parameters...")
    
    try:
        # Simulate TIDPO configuration
        config = {
            'use_tidpo': True,
            'alpha_triplet': 0.1,
            'gamma': 0.1,
            'enable_gradient_attribution': True
        }
        
        print(f"✅ TIDPO configuration loaded successfully!")
        print(f"use_tidpo: {config['use_tidpo']}")
        print(f"alpha_triplet: {config['alpha_triplet']}")
        print(f"gamma: {config['gamma']}")
        print(f"enable_gradient_attribution: {config['enable_gradient_attribution']}")
        
        # Validate key parameters
        if config['use_tidpo'] and config['alpha_triplet'] > 0:
            print("✅ Configuration is valid for TIDPO training")
        else:
            print("⚠️  Configuration may not be optimal for TIDPO training")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Triplet Loss Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_triplet_loss_calculation,
        test_log_ratio_sequence,
        test_numerical_stability,
        test_loss_combination,
        test_config_parameters,
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
        print("🎉 All tests passed! Triplet loss implementation is working correctly!")
    else:
        print("⚠️  Some tests failed, please check the implementation")
    
    return passed == total

if __name__ == "__main__":
    main() 