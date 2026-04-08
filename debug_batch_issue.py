#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Debug script for empty batch issue
# 
# This script helps debug issues with empty batches in data iteration

import sys
import os
import torch

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_data_iterator():
    """Debug data iterator"""
    print("🔍 Debugging data iterator...")
    
    try:
        from preference_datasets import get_batch_iterator
        from transformers import AutoTokenizer
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test training data iterator
        print("Testing training data iterator...")
        train_iterator = get_batch_iterator(
            names=['hh'],
            tokenizer=tokenizer,
            split='train',
            batch_size=1,  # Use small batch size to trigger the issue
            n_examples=10,
            max_length=128,
            max_prompt_length=64,
            silent=False  # Enable verbose output
        )
        
        batch_count = 0
        for batch in train_iterator:
            batch_count += 1
            print(f"Training batch {batch_count}:")
            print(f"  chosen_input_ids shape: {batch['chosen_input_ids'].shape}")
            print(f"  rejected_input_ids shape: {batch['rejected_input_ids'].shape}")
            
            # Check for empty tensors
            if batch['chosen_input_ids'].shape[0] == 0 or batch['rejected_input_ids'].shape[0] == 0:
                print(f"❌ Batch {batch_count} contains empty tensors!")
                print(f"  chosen_input_ids: {batch['chosen_input_ids']}")
                print(f"  rejected_input_ids: {batch['rejected_input_ids']}")
                return False
            
            if batch_count >= 5:  # Only check first 5 batches
                break
        
        print(f"✅ Checked {batch_count} training batches")
        
        # Test evaluation data iterator
        print("\nTesting evaluation data iterator...")
        eval_iterator = get_batch_iterator(
            names=['hh'],
            tokenizer=tokenizer,
            split='test',
            batch_size=1,
            n_examples=5,
            max_length=128,
            max_prompt_length=64,
            silent=False
        )
        
        batch_count = 0
        for batch in eval_iterator:
            batch_count += 1
            print(f"Evaluation batch {batch_count}:")
            print(f"  chosen_input_ids shape: {batch['chosen_input_ids'].shape}")
            print(f"  rejected_input_ids shape: {batch['rejected_input_ids'].shape}")
            
            if batch['chosen_input_ids'].shape[0] == 0 or batch['rejected_input_ids'].shape[0] == 0:
                print(f"❌ Evaluation batch {batch_count} contains empty tensors!")
                return False
            
            if batch_count >= 3:  # Only check first 3 batches
                break
        
        print(f"✅ Checked {batch_count} evaluation batches")
        return True
        
    except Exception as e:
        print(f"❌ Data iterator debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_collate_fn():
    """Debug collate_fn function"""
    print("\n🔍 Debugging collate_fn function...")
    
    try:
        from preference_datasets import get_collate_fn
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        collate_fn = get_collate_fn(tokenizer)
        
        # Create test batch
        test_batch = [
            {
                'chosen_input_ids': [1, 2, 3, 4, 5],
                'rejected_input_ids': [1, 2, 3, 4, 6],
                'chosen_attention_mask': [1, 1, 1, 1, 1],
                'rejected_attention_mask': [1, 1, 1, 1, 1],
                'chosen_labels': [1, 2, 3, 4, 5],
                'rejected_labels': [1, 2, 3, 4, 6],
                'prompt': "test prompt"
            }
        ]
        
        print("Testing normal batch...")
        result = collate_fn(test_batch)
        print(f"✅ Normal batch test passed")
        print(f"  chosen_input_ids shape: {result['chosen_input_ids'].shape}")
        print(f"  rejected_input_ids shape: {result['rejected_input_ids'].shape}")
        
        # Test empty batch
        print("\nTesting empty batch...")
        try:
            empty_batch = []
            result = collate_fn(empty_batch)
            print("❌ Empty batch test failed - should raise an exception")
            return False
        except ValueError as e:
            print(f"✅ Empty batch test passed - correct exception raised: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ collate_fn debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def debug_tokenize_batch_element():
    """Debug tokenize_batch_element function"""
    print("\n🔍 Debugging tokenize_batch_element function...")
    
    try:
        from preference_datasets import tokenize_batch_element
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test normal sample
        print("Testing normal sample...")
        result = tokenize_batch_element(
            prompt="Hello, how are you?",
            chosen="I am doing well, thank you.",
            rejected="I am not doing well.",
            truncation_mode="keep_end",
            tokenizer=tokenizer,
            max_length=128,
            max_prompt_length=64
        )
        
        print(f"✅ Normal sample test passed")
        print(f"  chosen_input_ids length: {len(result['chosen_input_ids'])}")
        print(f"  rejected_input_ids length: {len(result['rejected_input_ids'])}")
        
        # Test empty string
        print("\nTesting empty string...")
        try:
            result = tokenize_batch_element(
                prompt="",
                chosen="",
                rejected="",
                truncation_mode="keep_end",
                tokenizer=tokenizer,
                max_length=128,
                max_prompt_length=64
            )
            print(f"✅ Empty string test passed")
            print(f"  chosen_input_ids length: {len(result['chosen_input_ids'])}")
            print(f"  rejected_input_ids length: {len(result['rejected_input_ids'])}")
        except Exception as e:
            print(f"❌ Empty string test failed: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ tokenize_batch_element debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main debug function"""
    print("🚀 Empty Batch Issue Debug")
    print("=" * 50)
    
    # Run all debugs
    debug_functions = [
        debug_tokenize_batch_element,
        debug_collate_fn,
        debug_data_iterator,
    ]
    
    results = []
    for debug_func in debug_functions:
        try:
            result = debug_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Debug execution failed: {e}")
            results.append(False)
    
    # Summarize results
    print("\n" + "=" * 50)
    print("📊 Debug Result Summary:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed debugs: {passed}/{total}")
    
    if passed == total:
        print("🎉 All debugs passed! No obvious issues found")
    else:
        print("⚠️  Issues found, please check the above output")
    
    return passed == total

if __name__ == "__main__":
    main() 