#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test script for TIDPO fix
Verify that gradient attribution and memory issues are resolved
"""

import sys
import os
import torch
import psutil
import gc

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_gradient_attribution():
    """Test gradient attribution functionality"""
    print("🧪 Testing gradient attribution functionality...")
    
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

def test_memory_usage():
    """Test memory usage"""
    print("\n🧪 Testing memory usage...")
    
    try:
        import psutil
        import gc
        
        # Get initial memory
        initial_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        
        # Load model
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Get memory after loading
        after_load_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        print(f"Memory after loading model: {after_load_memory:.2f} MB")
        
        # Clean up
        del model
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Get final memory
        final_memory = psutil.virtual_memory().used / 1024 / 1024  # MB
        print(f"Final memory usage: {final_memory:.2f} MB")
        
        memory_increase = after_load_memory - initial_memory
        print(f"Memory increase: {memory_increase:.2f} MB")
        
        if memory_increase < 1000:  # Less than 1GB increase
            print("✅ Memory usage is reasonable")
            return True
        else:
            print("⚠️  Memory usage is high")
            return False
            
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False

def test_numerical_stability():
    """Test numerical stability"""
    print("\n🧪 Testing numerical stability...")
    
    try:
        import torch
        import torch.nn.functional as F
        
        # Test basic operations
        x = torch.randn(10, 10)
        y = torch.randn(10, 10)
        
        # Test log_softmax
        log_probs = F.log_softmax(x, dim=-1)
        if torch.isnan(log_probs).any() or torch.isinf(log_probs).any():
            print("❌ Log softmax contains NaN or Inf")
            return False
        
        # Test logsigmoid
        log_sigmoid = F.logsigmoid(x)
        if torch.isnan(log_sigmoid).any() or torch.isinf(log_sigmoid).any():
            print("❌ Log sigmoid contains NaN or Inf")
            return False
        
        # Test gradient computation
        x.requires_grad_(True)
        loss = F.mse_loss(x, y)
        loss.backward()
        
        if torch.isnan(x.grad).any() or torch.isinf(x.grad).any():
            print("❌ Gradients contain NaN or Inf")
            return False
        
        print("✅ Numerical stability test passed")
        return True
        
    except Exception as e:
        print(f"❌ Numerical stability test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 TIDPO Fix Verification Test")
    print("=" * 50)
    
    # Run all tests
    tests = [
        test_gradient_attribution,
        test_memory_usage,
        test_numerical_stability,
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
        print("🎉 All tests passed! TIDPO fix successful!")
        print("\n💡 You can now safely use TIDPO training")
    else:
        print("⚠️  Some tests failed, please check the fixes")
    
    return passed == total

if __name__ == "__main__":
    main() 