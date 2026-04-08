# Troubleshooting Guide

This guide provides solutions for common issues encountered when using the Token Importance Direct Preference Optimization (TIDPO) implementation.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Memory Problems](#memory-problems)
- [Training Issues](#training-issues)
- [Gradient Attribution Problems](#gradient-attribution-problems)
- [Configuration Issues](#configuration-issues)
- [Performance Issues](#performance-issues)
- [Debugging Tools](#debugging-tools)

## Installation Issues

### Issue: ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'gradient_attribution'
```

**Solutions:**

1. **Check Python path:**
   ```bash
   # Add current directory to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

2. **Verify installation:**
   ```bash
   python -c "import gradient_attribution; print('✅ Module loaded successfully')"
   ```

3. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt --force-reinstall
   ```

### Issue: CUDA Version Mismatch

**Symptoms:**
```
RuntimeError: CUDA version mismatch
```

**Solutions:**

1. **Check CUDA version:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.version.cuda)"
   ```

2. **Reinstall PyTorch with correct CUDA version:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## Memory Problems

### Issue: CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce batch size:**
   ```yaml
   # config/config.yaml
   batch_size: 2
   eval_batch_size: 2
   ```

2. **Enable gradient checkpointing:**
   ```yaml
   activation_checkpointing: true
   ```

3. **Use memory-optimized configuration:**
   ```bash
   python train.py config=config/config_memory_optimized.yaml
   ```

4. **Increase gradient accumulation:**
   ```yaml
   gradient_accumulation_steps: 4
   ```

5. **Reduce sequence length:**
   ```yaml
   max_length: 128
   max_prompt_length: 64
   ```

6. **Use mixed precision:**
   ```yaml
   policy_dtype: float16
   reference_dtype: float16
   ```

### Issue: System Memory Exhaustion

**Symptoms:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Clear cache:**
   ```python
   import gc
   import torch
   
   gc.collect()
   torch.cuda.empty_cache()
   ```

2. **Reduce model size:**
   ```yaml
   model: gpt2_small  # Use smaller model
   ```

3. **Use CPU for some operations:**
   ```python
   # Move some tensors to CPU
   tensor = tensor.cpu()
   ```

## Training Issues

### Issue: NaN Loss Values

**Symptoms:**
```
Loss becomes NaN during training
```

**Solutions:**

1. **Use float32 precision:**
   ```yaml
   policy_dtype: float32
   reference_dtype: float32
   ```

2. **Reduce learning rate:**
   ```yaml
   lr: 1e-6
   ```

3. **Enable gradient clipping:**
   ```yaml
   max_grad_norm: 1.0
   ```

4. **Check data quality:**
   ```python
   # Add data validation
   if torch.isnan(batch['chosen_input_ids']).any():
       print("Warning: NaN values in input data")
   ```

5. **Use gradient clipping in code:**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

### Issue: Loss Not Converging

**Symptoms:**
```
Loss remains high or oscillates
```

**Solutions:**

1. **Adjust learning rate:**
   ```yaml
   lr: 5e-6  # Try different learning rates
   ```

2. **Increase training epochs:**
   ```yaml
   n_epochs: 3
   ```

3. **Check data preprocessing:**
   ```bash
   python debug_batch_issue.py
   ```

4. **Monitor gradients:**
   ```python
   # Add gradient monitoring
   for name, param in model.named_parameters():
       if param.grad is not None:
           grad_norm = param.grad.norm()
           if grad_norm > 10:
               print(f"Large gradient in {name}: {grad_norm}")
   ```

### Issue: Empty Batches

**Symptoms:**
```
RuntimeError: cannot reshape tensor of 0 elements
```

**Solutions:**

1. **Increase batch size:**
   ```yaml
   batch_size: 4
   ```

2. **Check data loading:**
   ```bash
   python test_batch_size_fix.py
   ```

3. **Verify dataset:**
   ```python
   # Check dataset size
   dataset = load_dataset("Anthropic/hh-rlhf")
   print(f"Dataset size: {len(dataset['train'])}")
   ```

## Gradient Attribution Problems

### Issue: Gradient Attribution Failures

**Symptoms:**
```
RuntimeError: can't retain_grad on Tensor that has requires_grad=False
```

**Solutions:**

1. **Ensure model supports inputs_embeds:**
   ```python
   # Check if model supports inputs_embeds
   hasattr(model, 'get_input_embeddings')
   ```

2. **Set model to training mode:**
   ```python
   model.train()
   ```

3. **Enable gradient computation:**
   ```python
   for param in model.parameters():
       param.requires_grad_(True)
   ```

4. **Check text length:**
   ```python
   # Limit text length
   max_length = 512
   if len(text) > max_length:
       text = text[:max_length]
   ```

### Issue: Gradient Attribution Returns Uniform Weights

**Symptoms:**
```
All token importance scores are equal
```

**Solutions:**

1. **Check gradient computation:**
   ```python
   # Verify gradients are computed
   if embeddings.grad is None:
       print("Warning: No gradients computed")
   ```

2. **Use different norm:**
   ```python
   # Try L2 norm instead of L1
   token_importances = torch.norm(grads, p=2, dim=1)
   ```

3. **Check model architecture:**
   ```python
   # Ensure model supports gradient attribution
   print(f"Model type: {type(model)}")
   ```

## Configuration Issues

### Issue: Configuration Not Found

**Symptoms:**
```
FileNotFoundError: config/loss/tidpo.yaml
```

**Solutions:**

1. **Check file paths:**
   ```bash
   ls config/loss/
   ls config/model/
   ```

2. **Create missing configuration:**
   ```yaml
   # config/loss/tidpo.yaml
   name: tidpo
   use_tidpo: true
   alpha_triplet: 0.2
   gamma: 0.1
   enable_gradient_attribution: true
   ```

3. **Use default configuration:**
   ```bash
   python train.py loss=sft  # Use SFT instead of TIDPO
   ```

### Issue: Invalid Configuration Parameters

**Symptoms:**
```
ValueError: Invalid configuration parameter
```

**Solutions:**

1. **Validate configuration:**
   ```python
   from omegaconf import OmegaConf
   
   config = OmegaConf.load("config/config.yaml")
   print(OmegaConf.to_yaml(config))
   ```

2. **Check parameter types:**
   ```yaml
   # Ensure correct types
   batch_size: 4  # Integer
   lr: 1e-5       # Float
   use_tidpo: true  # Boolean
   ```

## Performance Issues

### Issue: Slow Training

**Symptoms:**
```
Training is very slow
```

**Solutions:**

1. **Use GPU acceleration:**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model.to(device)
   ```

2. **Enable mixed precision:**
   ```yaml
   policy_dtype: float16
   reference_dtype: float16
   ```

3. **Increase batch size:**
   ```yaml
   batch_size: 8
   ```

4. **Use gradient accumulation:**
   ```yaml
   gradient_accumulation_steps: 2
   ```

### Issue: High Memory Usage

**Symptoms:**
```
Memory usage is very high
```

**Solutions:**

1. **Enable gradient checkpointing:**
   ```yaml
   activation_checkpointing: true
   ```

2. **Reduce sequence length:**
   ```yaml
   max_length: 128
   max_prompt_length: 64
   ```

3. **Use smaller model:**
   ```yaml
   model: gpt2_small
   ```

4. **Clear cache regularly:**
   ```python
   # Add to training loop
   if step % 100 == 0:
       torch.cuda.empty_cache()
   ```

## Debugging Tools

### Debug Scripts

1. **Test gradient attribution:**
   ```bash
   python test_gradient_attribution.py
   ```

2. **Test TIDPO functionality:**
   ```bash
   python test_tidpo.py
   ```

3. **Test batch processing:**
   ```bash
   python test_batch_size_fix.py
   ```

4. **Debug batch issues:**
   ```bash
   python debug_batch_issue.py
   ```

### Debug Mode

Enable debug mode for detailed output:

```bash
python train.py debug=true
```

### Logging

Add logging to track issues:

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add to training loop
logger.debug(f"Batch shape: {batch['chosen_input_ids'].shape}")
logger.debug(f"Loss value: {loss.item()}")
```

### Memory Monitoring

Monitor memory usage:

```python
import psutil
import torch

def print_memory_usage():
    print(f"CPU Memory: {psutil.virtual_memory().percent}%")
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
```

### Common Debug Commands

```bash
# Check GPU status
nvidia-smi

# Monitor training logs
tail -f .cache/experiment_name_*/train.log

# Check Python environment
python -c "import torch; print(torch.__version__)"

# Test imports
python -c "import gradient_attribution, trainers, preference_datasets; print('All modules imported successfully')"
```

## Getting Help

If you encounter issues not covered in this guide:

1. **Check the logs:** Look at training logs for error messages
2. **Run debug scripts:** Use the provided debug scripts
3. **Check GitHub issues:** Search for similar issues
4. **Create minimal example:** Reproduce the issue with minimal code
5. **Provide system info:** Include Python version, PyTorch version, CUDA version

### System Information

When reporting issues, include:

```bash
# Python and package versions
python --version
pip list | grep torch
pip list | grep transformers

# GPU information
nvidia-smi

# System information
uname -a
```

### Example Issue Report

```
Issue: CUDA out of memory during training

System:
- Python 3.8.10
- PyTorch 1.12.1+cu113
- GPU: RTX 3080 (10GB)

Configuration:
- batch_size: 4
- max_length: 256
- model: gpt2_small

Error:
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB

Steps to reproduce:
1. Run: python train.py model=gpt2_small datasets=[hh] loss=tidpo
2. Error occurs after 100 steps

Attempted solutions:
- Reduced batch_size to 2
- Enabled activation_checkpointing
- Still getting OOM error
```

This troubleshooting guide should help resolve most common issues. For additional support, refer to the main README and API documentation. 