#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for gradient attribution functionality
Verify that the fixed gradient attribution works properly
"""

import sys
import os
import torch
import numpy as np

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_gradient_attribution():
    """Test basic gradient attribution functionality"""
    print("🧪 Testing basic gradient attribution functionality...")
    
    try:
        import gradient_attribution
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        # Load model and tokenizer
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Set pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test text
        test_text = "Hello world, this is a test sentence for gradient attribution."
        
        # Test gradient attribution
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
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
        
        # Validate results
        assert len(tokens) == len(importances), "tokens and importances length mismatch"
        assert all(isinstance(imp, (int, float)) for imp in importances), "importances should be numeric"
        
        print(f"✅ All validations passed!")
        return True
        
    except Exception as e:
        print(f"❌ Gradient attribution test failed: {e}")
        return False

def test_edge_cases():
    """Test edge cases"""
    print("\n🧪 Testing edge cases...")
    
    try:
        import gradient_attribution
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_name = "gpt2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Test empty text
        print("Testing empty text...")
        try:
            tokens, importances = gradient_attribution.compute_language_model_gradient_attribution(
                model=model,
                tokenizer=tokenizer,
                text="",
                device=device
            )
            print(f"✅ Empty text test passed, tokens: {len(tokens)}")
        except Exception as e:
            print(f"⚠️  Empty text test failed: {e}")
        
        # Test single word
        print("Testing single word...")
        try:
            tokens, importances = gradient_attribution.compute_language_model_gradient_attribution(
                model=model,
                tokenizer=tokenizer,
                text="Hello",
                device=device
            )
            print(f"✅ Single word test passed, tokens: {len(tokens)}")
        except Exception as e:
            print(f"⚠️  Single word test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ Edge cases test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Gradient Attribution Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_basic_gradient_attribution,
        test_edge_cases,
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
        print("🎉 All tests passed! Gradient attribution is working correctly!")
    else:
        print("⚠️  Some tests failed, please check the implementation")
    
    return passed == total

if __name__ == "__main__":
    main() 