# Quick Start Guide

This guide will help you get started with Token Importance Direct Preference Optimization (TIDPO) in just a few minutes.

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash

# Install dependencies
pip install -r requirements.txt

# Set up environment
python setup_environment.py
```

### Step 2: Verify Installation

```bash
# Test gradient attribution
python test_gradient_attribution.py

# Test TIDPO functionality
python test_tidpo.py
```

### Step 3: Run Your First Training

```bash
# Use the example script (recommended for beginners)
python run_tidpo_example.py
```

This will:
- Perform supervised fine-tuning (SFT)
- Run TIDPO training with gradient attribution
- Save results to `.cache/` directory

## 📚 Understanding the Basics



### What is TIDPO?

TIDPO (Token Importance DPO) adds gradient attribution to DPO for enhanced training.

**Key Features:**
- **Gradient attribution**: Compute token importance using gradients
- **Weighted loss**: Apply importance weights to loss function
- **Triplet loss**: Additional loss for better training

**Mathematical Foundation:**

TIDPO combines weighted DPO loss with triplet loss:

```
L_TIDPO = -log σ(β * Σ_t w_t * [log π_θ(y_t) - log π_ref(y_t)] - α * δ)
L_triplet = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
L_total = L_TIDPO + α_triplet * L_triplet
```

Where:
- `w_t`: Token importance weights from gradient attribution
- `α_triplet`: Triplet loss weight (default: 0.2)
- `margin`: Distance margin (default: 0.2)

## 🎯 Common Use Cases

### Case 1: Quick Experiment

```bash
# Simple training with default settings
python train.py \
    model=gpt2_small \
    datasets=[hh] \
    loss=tidpo \
    exp_name=my_first_experiment \
    batch_size=4 \
    n_epochs=1
```

### Case 2: Memory-Constrained Environment

```bash
# Use memory-optimized configuration
python train.py \
    config=config/config_memory_optimized.yaml \
    exp_name=memory_efficient_training
```

### Case 3: Research Experiment

```bash
# Full training pipeline
python train.py \
    model=gpt2_small \
    datasets=[hh] \
    loss=tidpo \
    exp_name=research_experiment \
    batch_size=4 \
    n_epochs=3 \
    lr=1e-5 \
    max_length=256 \
    gradient_accumulation_steps=2 \
    activation_checkpointing=true
```

## ⚙️ Configuration Quick Reference

### Basic Configuration

```yaml
# config/config.yaml
seed: 0
exp_name: my_experiment
batch_size: 4
eval_batch_size: 4
lr: 1e-5
n_epochs: 1
model: gpt2_small
loss: tidpo
datasets: [hh]
max_length: 256
max_prompt_length: 128
gradient_accumulation_steps: 1
activation_checkpointing: true
```

### TIDPO Configuration

```yaml
# config/loss/tidpo.yaml
name: tidpo
use_tidpo: true
alpha_triplet: 0.2
gamma: 0.1
enable_gradient_attribution: true
alpha: 0.5
beta: 0.1
```

### Model Configuration

```yaml
# config/model/gpt2_small.yaml
name_or_path: gpt2
policy_dtype: float32
reference_dtype: float32
```

## 🔧 Essential Commands

### Training Commands

```bash
# Basic training
python train.py model=gpt2_small datasets=[hh] loss=tidpo

# With custom configuration
python train.py config=config/config_memory_optimized.yaml

# Debug mode
python train.py debug=true

# Custom experiment name
python train.py exp_name=my_custom_experiment
```

### Testing Commands

```bash
# Test all components
python test_gradient_attribution.py
python test_tidpo.py
python test_triplet_loss.py
python test_batch_size_fix.py

# Debug specific issues
python debug_batch_issue.py
```

### Monitoring Commands

```bash
# Monitor training logs
tail -f .cache/experiment_name_*/train.log

# Check GPU usage
nvidia-smi -l 1

# Monitor memory
python -c "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB')"
```

## 📊 Understanding Outputs

### Training Logs

Training logs show:
- **Loss values**: Weighted DPO loss, triplet loss, total loss
- **Metrics**: Preference accuracy, KL divergence
- **Memory usage**: GPU memory consumption
- **Progress**: Steps completed, time elapsed

### Model Outputs

Models are saved to:
```
.cache/experiment_name_*/LATEST/
├── policy.pt          # Trained policy model
├── reference.pt       # Reference model
├── tokenizer/         # Tokenizer files
└── train.log         # Training logs
```

### Evaluation Results

Evaluation shows:
- **Preference accuracy**: How well the model follows preferences
- **KL divergence**: Distance from reference model
- **Generation quality**: Sample outputs

## 🛠️ Common Customizations

### Custom Token Importance

```python
# Modify gradient attribution
from gradient_attribution import compute_language_model_gradient_attribution

def custom_importance(model, tokenizer, text, device):
    tokens, importances = compute_language_model_gradient_attribution(
        model, tokenizer, text, device
    )
    # Apply custom logic
    modified_importances = [imp * 0.8 + 0.2 for imp in importances]
    return tokens, modified_importances
```

### Custom Loss Function

```python
# Modify loss calculation
def custom_tdpo_loss(chosen_logps, rejected_logps, beta=0.1):
    logits = beta * (chosen_logps - rejected_logps)
    losses = -F.logsigmoid(logits)
    return losses.mean()
```

### Custom Dataset

```python
# Add custom dataset
def load_custom_dataset():
    return [
        {
            'prompt': 'Your prompt here',
            'chosen': 'Preferred response',
            'rejected': 'Non-preferred response'
        }
    ]
```

## 🚨 Quick Troubleshooting

### Memory Issues

```bash
# Reduce batch size
python train.py batch_size=2

# Enable memory optimizations
python train.py activation_checkpointing=true gradient_accumulation_steps=4

# Use memory-optimized config
python train.py config=config/config_memory_optimized.yaml
```

### Training Issues

```bash
# Use float32 for stability
python train.py policy_dtype=float32 reference_dtype=float32

# Reduce learning rate
python train.py lr=1e-6

# Enable gradient clipping
python train.py max_grad_norm=1.0
```

### Gradient Attribution Issues

```bash
# Test gradient attribution
python test_gradient_attribution.py

# Use smaller model
python train.py model=gpt2_small

# Check model compatibility
python -c "from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('gpt2'); print(hasattr(model, 'get_input_embeddings'))"
```

## 📈 Next Steps

### For Beginners

1. **Run the example script**: `python run_tidpo_example.py`
2. **Experiment with parameters**: Try different batch sizes, learning rates
3. **Monitor training**: Watch logs and GPU usage
4. **Test components**: Run individual test scripts

### For Researchers

1. **Read the paper**: Understand the theoretical foundations
2. **Study the code**: Examine `trainers.py` and `gradient_attribution.py`
3. **Modify algorithms**: Implement custom loss functions
4. **Add datasets**: Integrate new preference datasets

### For Developers

1. **Extend functionality**: Add new model architectures
2. **Optimize performance**: Improve memory efficiency
3. **Add tests**: Create comprehensive test suites
4. **Documentation**: Improve API documentation

## 📚 Additional Resources

### Documentation

- [README.md](README.md): Comprehensive project overview
- [API_DOCUMENTATION.md](API_DOCUMENTATION.md): Detailed API reference
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md): Common issues and solutions

### Code Examples

- [run_tidpo_example.py](run_tidpo_example.py): Complete training example
- [test_*.py](test_*.py): Test scripts for all components
- [debug_*.py](debug_*.py): Debugging tools

### Configuration Files

- [config/config.yaml](config/config.yaml): Main configuration
- [config/loss/tidpo.yaml](config/loss/tidpo.yaml): TIDPO parameters
- [config/model/gpt2_small.yaml](config/model/gpt2_small.yaml): Model settings

## 🎉 Congratulations!

You've successfully set up and run TIDPO! 

**What you've accomplished:**
- ✅ Installed the TIDPO implementation
- ✅ Verified all components work correctly
- ✅ Run your first training experiment
- ✅ Learned the basic concepts and commands

**Next steps:**
- Experiment with different configurations
- Try different models and datasets
- Read the detailed documentation
- Contribute to the project

Happy training! 🚀 