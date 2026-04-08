#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for batch size fix
Verify that batch size of 1 does not produce empty tensors
"""

import sys
import os
import torch
import numpy as np

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_concatenated_inputs():
    """Test concatenated_inputs function"""
    print("🧪 Testing concatenated_inputs function...")
    
    try:
        from trainers import concatenated_inputs
        
        # Create test batch
        pad_token_id = 42
        batch = {
            'chosen_input_ids': torch.randint(0, 1000, (2, 10)),
            'rejected_input_ids': torch.randint(0, 1000, (2, 8)),
            'chosen_attention_mask': torch.ones(2, 10, dtype=torch.long),
            'rejected_attention_mask': torch.ones(2, 8, dtype=torch.long),
        }
        
        # Test normal case
        print("Testing normal batch...")
        result = concatenated_inputs(batch, pad_token_id=pad_token_id)
        print(f"✅ Normal batch test passed")
        print(f"Result shape: {result['concatenated_input_ids'].shape}")

        # Verify padding semantics: wherever attention_mask == 0, input_ids should equal pad_token_id
        if 'concatenated_attention_mask' in result:
            pad_positions = (result['concatenated_attention_mask'] == 0)
            if pad_positions.any():
                padded_ids = result['concatenated_input_ids'][pad_positions]
                if not torch.all(padded_ids == pad_token_id):
                    print("❌ Padding value mismatch: input_ids not padded with pad_token_id")
                    return False
        
        # Test empty batch (should raise exception)
        print("Testing empty batch...")
        empty_batch = {
            'chosen_input_ids': torch.empty(0, 0, dtype=torch.long),
            'rejected_input_ids': torch.empty(0, 0, dtype=torch.long),
            'chosen_attention_mask': torch.empty(0, 0, dtype=torch.long),
            'rejected_attention_mask': torch.empty(0, 0, dtype=torch.long)
        }
        
        try:
            result = concatenated_inputs(empty_batch, pad_token_id=pad_token_id)
            print("❌ Empty batch test failed - should raise exception")
            return False
        except Exception as e:
            print(f"✅ Empty batch test passed - correctly raised exception: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ concatenated_inputs test failed: {e}")
        return False

def test_batch_iterator():
    """Test batch iterator with small batch size"""
    print("\n🧪 Testing batch iterator with small batch size...")
    
    try:
        from preference_datasets import get_batch_iterator
        
        # Simulate dataset with small batch size
        dataset = [
            {'chosen': 'Hello world', 'rejected': 'Hi there'},
            {'chosen': 'Good morning', 'rejected': 'Morning'},
            {'chosen': 'How are you', 'rejected': 'How do you do'},
            {'chosen': 'Nice to meet you', 'rejected': 'Pleased to meet you'}
        ]
        
        # Test with batch_size=1
        batch_size = 1
        iterator = get_batch_iterator(dataset, batch_size)
        
        batch_count = 0
        for batch in iterator:
            batch_count += 1
            print(f"Batch {batch_count}: {batch['chosen_input_ids'].shape}")
            
            # Verify batch is not empty
            if batch['chosen_input_ids'].shape[0] == 0:
                print(f"❌ Batch {batch_count} is empty")
                return False
        
        print(f"✅ Batch iterator test passed, processed {batch_count} batches")
        return True
        
    except Exception as e:
        print(f"❌ Batch iterator test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Batch Size Fix Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_concatenated_inputs,
        test_batch_iterator,
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
        print("🎉 All tests passed! Batch size fix is working correctly!")
    else:
        print("⚠️  Some tests failed, please check the implementation")
    
    return passed == total

if __name__ == "__main__":
    main() 