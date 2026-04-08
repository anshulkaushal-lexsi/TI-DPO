# Token-importance Direct Preference Optimization (TIDPO)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-red.svg)](https://pytorch.org/)


## 🚀 Features

- **TIDPO Extension**: Token Importance DPO with gradient attribution
- **Gradient Attribution**: Advanced token importance calculation using gradient-based attribution
- **Memory Optimization**: Efficient memory usage with gradient checkpointing and mixed precision
- **Multiple Model Support**: Support for Mistral, Llama, GPT-2, Pythia, and other transformer models
- **Comprehensive Testing**: Extensive test suite for all components
- **Easy Configuration**: YAML-based configuration system

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Concepts](#core-concepts)
- [Usage](#usage)
- [Configuration](#configuration)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Citation](#citation)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import gradient_attribution; print('✅ Installation successful!')"
```

Note: make sure the `python` you run has PyTorch installed. On Windows with Anaconda, you may need to use your conda python explicitly (e.g., `D:\anaconda3\python.exe`) or activate the environment first.

### Environment Setup

```bash
# Set up environment variables and cache directories
python setup_environment.py
```

## 🚀 Quick Start

### Method 1: Use the Example Script (Recommended)

```bash
# Run the complete TIDPO training pipeline
python run_tidpo_example.py
```

This script will:
1. Perform supervised fine-tuning (SFT)
2. Run TIDPO training with gradient attribution
3. Save models and logs to `.cache/` directory

### Method 2: Manual Training

Tip (Windows/Anaconda): if `python` points to a system interpreter without PyTorch, replace `python` with your conda python (e.g., `D:\anaconda3\python.exe`).

#### Step 1: Supervised Fine-tuning (SFT)

```bash
python -u train.py \
    model=gpt2_small \
    datasets=[hh] \
    loss=sft \
    exp_name=my_experiment \
    batch_size=4 \
    eval_batch_size=4 \
    n_epochs=1 \
    lr=1e-5 \
    max_length=256 \
    max_prompt_length=128 \
    gradient_accumulation_steps=1 \
    activation_checkpointing=true
```

#### Step 2: TIDPO Training

```bash
python -u train.py \
    model=gpt2_small \
    datasets=[hh] \
    loss=tidpo \
    exp_name=my_experiment \
    batch_size=4 \
    eval_batch_size=4 \
    n_epochs=1 \
    lr=1e-5 \
    max_length=256 \
    max_prompt_length=128 \
    gradient_accumulation_steps=1 \
    activation_checkpointing=true
```

## 🧠 Core Concepts

### TIDPO Algorithm

In this codebase, TI-DPO is built on a TDPO-style *pairwise* objective (chosen vs rejected), with an additional position-wise KL correction term (TDPO2 by default).

Let the per-sequence log-ratio margin be:

```
m(y) = Σ_t [ log π_θ(y_t | x, y_{<t}) - log π_ref(y_t | x, y_{<t}) ]
```

and let `k(y)` denote the aggregated position-wise KL term used by TDPO (see `tdpo_loss` in `trainers.py`). Then the TDPO-style logistic objective is:

```
L_TDPO = -log σ( β * ( (m(y⁺) - m(y⁻)) - α * (k(y⁻) - stopgrad(k(y⁺))) ) )
```

where `y⁺` is the chosen response and `y⁻` is the rejected response.

### TIDPO Extension

TI-DPO introduces token-importance weights `w_t` (from gradient attribution) by reweighting the log-ratio sum:

```
m_w(y) = Σ_t w_t * [ log π_θ(y_t | x, y_{<t}) - log π_ref(y_t | x, y_{<t}) ]

L_TIDPO = -log σ( β * ( (m_w(y⁺) - m_w(y⁻)) - α * (k(y⁻) - stopgrad(k(y⁺))) ) )
```

Where `w_t` is the importance weight calculated using gradient attribution.

### Triplet Loss Component

TIDPO incorporates triplet loss to enhance training by learning better representations:

```
L_triplet = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
```

Where:
- `anchor`: An intermediate “anchor” response sampled from the *current policy model* on the same prompt (see `_generate_anchor_outputs` in `trainers.py`)
- `positive`: Chosen response (`y⁺`)
- `negative`: Rejected response (`y⁻`)
- `d(·,·)`: Mask-aware squared L2 distance between token-wise log-ratio vectors (`log π_θ - log π_ref`) of two responses
- `margin`: Triplet margin `loss.alpha_triplet` (set `> 0` to enable; default in configs may be `0.0`)

The complete TIDPO loss combines both components:

```
L_total = L_TIDPO + γ * L_triplet
```

Where `γ` controls the weight of triplet loss. In this codebase, `alpha_triplet` is the triplet *margin* used inside `L_triplet`.

### Gradient Attribution

The gradient attribution module calculates token importance by:
1. Computing gradients with respect to input embeddings
2. Using L1 norm for importance scoring
3. Normalizing scores for stable training
4. Applying mixed strategy with Gaussian prior for robustness

## 📖 Usage

### Training Pipeline

The complete training pipeline consists of two stages:

1. **Supervised Fine-tuning (SFT)**: Pre-train the model on preference data
2. **TIDPO Training**: Apply token importance preference optimization

### Configuration Files

Key configuration files:

- `config/config.yaml`: Main configuration
- `config/loss/tidpo.yaml`: TIDPO-specific parameters
- `config/model/gpt2_small.yaml`: Model configuration
- `config/config_memory_optimized.yaml`: Memory-optimized settings

### Available Models

- `gpt2_small`: GPT-2 small (124M parameters)
- `gpt2_large`: GPT-2 large (774M parameters)
- `pythia28`: Pythia-2.8B
- `pythia69`: Pythia-6.9B
- `llama7b`: LLaMA-7B
- `mistral7b`: Mistral-7B
- `mistral7b_instruct`: Mistral-7B-Instruct
- `llama3b`: LLaMA-3B

### Available Datasets

- `hh`: Anthropic Helpful-Harmless (HH) preference dataset (quick start)
- `shp`: Stanford Human Preferences (SHP) dataset (quick start)
- `se`: StackExchange preference dataset (quick start)

These built-in datasets are meant for getting started quickly. Other datasets/benchmarks (e.g., MMLU, TruthfulQA, GSM8K, MATH, MT-Bench, Arena-Hard, etc.) typically require manual download plus additional evaluation configuration (see the next section).

## 📈 Evaluation Benchmarks (Paper)

This repo currently focuses on preference-training datasets (e.g., `hh/shp/se`). Many paper benchmarks (MMLU, GSM8K, MATH, MT-Bench, Arena-Hard, TruthfulQA, GPQA, IFEval, HumanEval, etc.) are typically evaluated using *separate* benchmarking harnesses.

### Where to get the data

- **MMLU**: commonly accessed via Hugging Face datasets (search “mmlu”) or EleutherAI’s evaluation tooling.
- **GSM8K**: available on Hugging Face datasets (search “gsm8k”).
- **MATH**: available on Hugging Face datasets (search “hendrycks/math”).
- **TruthfulQA**: available on Hugging Face datasets (search “truthful_qa”).
- **HumanEval**: available via OpenAI’s HumanEval release and mirrors; also supported by common evaluation harnesses.
- **MT-Bench**: released by LMSYS; evaluation is usually run with their scripts and requires an LLM “judge” (or a specified judge model).
- **Arena-Hard**: typically provided/maintained by LMSYS; evaluation is usually done with their provided prompts + judge setup.
- **GPQA**: released by the authors; may have access/licensing restrictions depending on the version.
- **IFEval**: released by the authors; commonly distributed via Hugging Face datasets (search “ifeval”).

Notes:
- If a dataset isn’t directly included in this repo, the easiest path is usually `pip install datasets` and loading it from Hugging Face by name.
- Some benchmarks (MT-Bench/Arena-style) are *not* simple “download + accuracy”; they rely on a judge model and a specific scoring pipeline.

### Recommended evaluation harnesses

- **lm-evaluation-harness** (EleutherAI): broad coverage for MMLU/GSM8K/TruthfulQA/MATH-style multiple-choice and QA.
- **LMSYS MT-Bench / Arena tooling**: for chat-style judge-based evaluation.
- **HumanEval toolchains**: run code-generation evaluation in a sandboxed environment.

## ⚙️ Configuration

### Notes (Hyperparameters & Reproducibility)

- **Different scenarios need different configs**: optimal settings vary by dataset (e.g., `hh` vs `shp`), model size, sequence lengths, optimizer/batch size, and whether you enable gradient-attribution or triplet. Treat the provided YAMLs as *reasonable defaults* for quick sanity checks; paper-level results usually require a sweep over `beta`, `lambda_importance`, `prior_sigma_div`, and optionally `gamma/alpha_triplet`.
- **Seeds are not perfectly “fixed” in practice**: even when a seed is set, results can differ across runs due to GPU/CUDA nondeterminism, kernel selection, mixed precision/TF32, dataloader order, and distributed training details.
- **Generation has extra uncertainty**: evaluation that involves sampling/decoding may vary with tokenizer/model versions, hardware, and decoding settings.

### TIDPO Parameters

```yaml
# config/loss/tidpo.yaml
name: tidpo
use_tidpo: true              # Enable TIDPO
beta: 0.2                    # Temperature parameter
alpha: 0.5                   # TDPO-style objective parameter
if_tdpo2: true               # Use TDPO2 variant

enable_gradient_attribution: true  # Enable gradient attribution

lambda_importance: 0.2       # Token-importance strength
prior_sigma_div: 8.0         # Gaussian prior (larger = weaker prior)

gamma: 0.01                   # Triplet loss weight (set >0 to enable)
alpha_triplet: 0.01           # Triplet margin (set >0 to enable)
kl_coef: 0.0                 # Optional KL regularization
```

### Memory Optimization

For limited GPU memory:

```yaml
# config/config_memory_optimized.yaml
batch_size: 4
eval_batch_size: 4
max_length: 512
max_prompt_length: 256
gradient_accumulation_steps: 1
activation_checkpointing: true
```

### Training Parameters

Recommended settings:

| Parameter | SFT | TIDPO |
|-----------|-----|-------|
| Learning Rate | 1e-5 | 1e-5 |
| Batch Size | 4-16 | 4-16 |
| Epochs | 1 | 1-3 |
| Max Length | 256 | 256 |
| Gradient Accumulation | 1-4 | 1-4 |

## 🔬 Advanced Features

### Gradient Attribution

```python
from gradient_attribution import compute_language_model_gradient_attribution

# Calculate token importance
tokens, importances = compute_language_model_gradient_attribution(
    model=model,
    tokenizer=tokenizer,
    text="Your input text here",
    device=device
)
```

### Custom Token Importance

```python
def custom_importance_function(model, tokenizer, text, device):
    # Implement your custom importance calculation
    tokens, importances = compute_language_model_gradient_attribution(
        model, tokenizer, text, device
    )
    # Apply your custom logic
    return modified_importances
```

### Triplet Loss

TIDPO includes triplet loss for enhanced training:

```python
# Triplet loss is computed when both are > 0
gamma: 0.1          # Weight for triplet loss term
alpha_triplet: 0.2  # Triplet margin
```

##  Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest -q

# Test gradient attribution
python test_gradient_attribution.py

# Test TIDPO functionality
python test_tidpo.py

# Test triplet loss
python test_triplet_loss.py

# Test batch processing
python test_batch_size_fix.py

# Debug batch issues
python debug_batch_issue.py
```

## Monitoring and Debugging

### Training Logs

```bash
# Monitor training progress
tail -f .cache/your_experiment_name_*/train.log

# Check GPU usage
nvidia-smi -l 1
```

### Debug Mode

```bash
# Enable debug mode for detailed output
python -u train.py ... debug=true
```

### Common Issues

#### 1. Out of Memory (OOM)

**Symptoms**: CUDA out of memory errors

**Solutions**:
- Reduce batch size: `batch_size: 2`
- Enable gradient checkpointing: `activation_checkpointing: true`
- Use memory-optimized config: `config/config_memory_optimized.yaml`
- Increase gradient accumulation: `gradient_accumulation_steps: 4`

#### 2. Gradient Attribution Failures

**Symptoms**: "can't retain_grad on Tensor that has requires_grad=False"

**Solutions**:
- Ensure model supports `inputs_embeds`
- Check text length limits
- Verify model is in training mode

#### 3. NaN Loss Values

**Symptoms**: Loss becomes NaN during training

**Solutions**:
- Use `float32` precision: `policy_dtype: float32`
- Reduce learning rate: `lr: 1e-6`
- Enable gradient clipping: `max_grad_norm: 1.0`
- Check data quality

#### 4. Empty Batches

**Symptoms**: "cannot reshape tensor of 0 elements"

**Solutions**:
- Increase batch size: `batch_size: 4`
- Check data preprocessing
- Verify dataset loading

## 📊 Performance Optimization

### Memory Optimization

1. **Gradient Checkpointing**: Reduces memory usage by ~50%
2. **Mixed Precision**: Use `float16` for faster training
3. **Batch Size Tuning**: Balance memory and training stability
4. **Sequence Length**: Reduce `max_length` for memory constraints

### Computational Optimization

1. **Gradient Attribution Caching**: Cache importance scores
2. **Batch Processing**: Process multiple samples together
3. **Parallel Computation**: Use multiple GPUs if available

### Training Stability

1. **Learning Rate Scheduling**: Use warmup and decay
2. **Gradient Clipping**: Prevent gradient explosion
3. **Loss Monitoring**: Track loss values for stability

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Add tests for new functionality
5. Run the test suite: `python -m pytest tests/`
6. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest -q

# Run linting
flake8 .

# Run type checking
mypy .
```

## 📚 Citation

If you use the TIDPO system in your academic research, please cite the following paper:

```bibtex
@misc{yang2025tokenimportanceguideddirectpreference,
      title={Token-Importance Guided Direct Preference Optimization}, 
      author={Ning Yang and Hai Lin and Yibo Liu and Baoliang Tian and Guoqing Liu and Haijun Zhang},
      year={2025},
      eprint={2505.19653},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.19653}, 
}

https://arxiv.org/abs/2505.19653, Token-Importance Guided Direct Preference Optimization, HTML version: https://arxiv.org/html/2505.19653v1




## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Original DPO implementation by [Eric Mitchell](https://github.com/eric-mitchell/direct-preference-optimization)
- Hugging Face Transformers for model support
- Anthropic for the HH-RLHF dataset


---

**Note**: This is a research implementation. For production use, additional testing and optimization may be required.

