# API Documentation

This document provides detailed API documentation for the Token Importance Direct Preference Optimization (TIDPO) implementation.

## Table of Contents

- [Mathematical Foundations](#mathematical-foundations)
- [Core Modules](#core-modules)
- [Training Components](#training-components)
- [Gradient Attribution](#gradient-attribution)
- [Configuration](#configuration)
- [Utilities](#utilities)
- [Testing](#testing)

## Mathematical Foundations

### TIDPO Algorithm

TIDPO extends DPO to operate at the token level with importance weighting:

```
L_TIDPO = -log σ(β * Σ_t w_t * [log π_θ(y_t) - log π_ref(y_t)] - α * δ)
```

Where:
- `β`: Temperature parameter
- `α`: KL divergence weight
- `δ`: Sequence KL divergence difference
- `π_θ`: Policy model
- `π_ref`: Reference model
- `w_t`: Token importance weights from gradient attribution

### Triplet Loss Component

TIDPO incorporates triplet loss to enhance training by learning better representations:

```
L_triplet = max(d(anchor, positive) - d(anchor, negative) + margin, 0)
```

Where:
- `anchor`: Reference model outputs
- `positive`: Chosen responses
- `negative`: Rejected responses
- `d(·,·)`: Distance function (typically L2 norm)
- `margin`: Minimum distance margin (default: 0.2)

The complete TIDPO loss combines both components:

```
L_total = L_TIDPO + α_triplet * L_triplet
```

Where `α_triplet` controls the weight of triplet loss (default: 0.2).

### Gradient Attribution

The gradient attribution module calculates token importance by:
1. Computing gradients with respect to input embeddings
2. Using L1 norm for importance scoring
3. Normalizing scores for stable training
4. Applying mixed strategy with Gaussian prior for robustness

## Core Modules

### `trainers.py`

The main training module containing the `BasicTrainer` class and related functions.

#### `BasicTrainer`

Main trainer class for TIDPO training.

```python
class BasicTrainer:
    def __init__(self, policy, config, seed, run_dir, reference_model=None):
        """
        Initialize the trainer.
        
        Args:
            policy: The policy model to train
            config: Training configuration
            seed: Random seed
            run_dir: Directory to save results
            reference_model: Reference model for KL divergence
        """
```

**Key Methods:**

- `train()`: Main training loop
- `eval()`: Evaluation loop
- `tdpo_concatenated_forward()`: TIDPO forward pass
- `_compute_token_importance_weights()`: Calculate token importance
- `_compute_triplet_loss()`: Compute triplet loss
- `_get_log_ratio_sequence()`: Calculate log-ratio sequences

#### `tdpo_concatenated_forward()`

```python
def tdpo_concatenated_forward(self, model, reference_model, batch):
    """
    Perform TIDPO forward pass with concatenated inputs.
    
    Args:
        model: Policy model
        reference_model: Reference model
        batch: Input batch
        
    Returns:
        Tuple of (chosen_logps_margin, rejected_logps_margin, 
                 chosen_position_kl, rejected_position_kl,
                 chosen_logps, rejected_logps, triplet_loss)
    """
```

#### `_compute_token_importance_weights()`

```python
def _compute_token_importance_weights(self, model, input_ids, attention_mask):
    """
    Compute token importance weights using gradient attribution.
    
    Args:
        model: Language model
        input_ids: Input token IDs
        attention_mask: Attention mask
        
    Returns:
        weight_matrix: Token importance weights [batch_size, seq_len]
    """
```

### `gradient_attribution.py`

Module for computing gradient-based token importance.

#### `compute_language_model_gradient_attribution()`

```python
def compute_language_model_gradient_attribution(
    model, tokenizer, text, device=None, target_label=None
):
    """
    Perform gradient attribution on language model.
    
    Args:
        model: Pre-trained language model
        tokenizer: Corresponding tokenizer
        text: Input text to be attributed
        device: torch.device, default CPU
        target_label: Optional target label for attribution
        
    Returns:
        tokens: List of tokenized tokens
        importances: List of importance scores
    """
```

**Example Usage:**

```python
from gradient_attribution import compute_language_model_gradient_attribution
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Compute token importance
tokens, importances = compute_language_model_gradient_attribution(
    model=model,
    tokenizer=tokenizer,
    text="Hello world, this is a test sentence.",
    device="cuda"
)

print(f"Tokens: {tokens}")
print(f"Importances: {importances}")
```

### `preference_datasets.py`

Module for loading and processing preference datasets.

#### `get_batch_iterator()`

```python
def get_batch_iterator(
    names, tokenizer, split, batch_size, n_examples=None,
    max_length=512, max_prompt_length=128, silent=False
):
    """
    Create batch iterator for preference datasets.
    
    Args:
        names: List of dataset names
        tokenizer: Tokenizer for text processing
        split: Dataset split ('train', 'test', 'validation')
        batch_size: Batch size
        n_examples: Number of examples to use
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        silent: Whether to suppress output
        
    Yields:
        batch: Dictionary containing tokenized inputs
    """
```

#### `tokenize_batch_element()`

```python
def tokenize_batch_element(
    prompt, chosen, rejected, tokenizer, max_length, max_prompt_length
):
    """
    Tokenize a single batch element.
    
    Args:
        prompt: Input prompt
        chosen: Chosen response
        rejected: Rejected response
        tokenizer: Tokenizer
        max_length: Maximum sequence length
        max_prompt_length: Maximum prompt length
        
    Returns:
        Dictionary with tokenized inputs
    """
```

## Training Components

### Loss Functions

#### `tdpo_loss()`

```python
def tdpo_loss(chosen_logps_margin, rejected_logps_margin, 
              chosen_position_kl, rejected_position_kl,
              beta=0.1, alpha=0.5, if_tdpo2=False):
    """
    Compute TIDPO loss.
    
    Args:
        chosen_logps_margin: Log probabilities for chosen responses
        rejected_logps_margin: Log probabilities for rejected responses
        chosen_position_kl: KL divergence for chosen responses
        rejected_position_kl: KL divergence for rejected responses
        beta: Temperature parameter
        alpha: KL divergence weight
        if_tdpo2: Whether to use TDPO2 variant (base algorithm)
        
    Returns:
        losses: TIDPO losses
        chosen_rewards: Rewards for chosen responses
        rejected_rewards: Rewards for rejected responses
    ```

#### `_weighted_tdpo_get_batch_logps()`

```python
def _weighted_tdpo_get_batch_logps(
    logits, reference_logits, labels, weight_matrix, average_log_prob=False
):
    """
    Compute weighted TIDPO log probabilities.
    
    Args:
        logits: Policy model logits
        reference_logits: Reference model logits
        labels: Target labels
        weight_matrix: Token importance weights
        average_log_prob: Whether to average log probabilities
        
    Returns:
        logps_margin: Weighted log probability differences
        position_kl: Position-wise KL divergence
        logps: Log probabilities
    """
```

### Triplet Loss

#### `_compute_triplet_loss()`

```python
def _compute_triplet_loss(self, model, reference_model, batch):
    """
    Compute triplet loss for TIDPO.
    
    Args:
        model: Policy model
        reference_model: Reference model
        batch: Input batch
        
    Returns:
        triplet_loss: Computed triplet loss
    """
```

## Configuration

### Configuration Files

#### `config/config.yaml`

Main configuration file with training parameters:

```yaml
# Basic training parameters
seed: 0
exp_name: my_experiment
batch_size: 4
eval_batch_size: 4
lr: 1e-5
n_epochs: 1

# Model configuration
model: gpt2_small
loss: tidpo

# Dataset configuration
datasets: [hh]
max_length: 256
max_prompt_length: 128

# Optimization
gradient_accumulation_steps: 1
activation_checkpointing: true
max_grad_norm: 1.0
```

#### `config/loss/tidpo.yaml`

TIDPO-specific configuration:

```yaml
name: tidpo
use_tidpo: true
alpha_triplet: 0.2
gamma: 0.1
enable_gradient_attribution: true
alpha: 0.5
beta: 0.1
```

#### `config/model/gpt2_small.yaml`

Model-specific configuration:

```yaml
name_or_path: gpt2
policy_dtype: float32
reference_dtype: float32
```

### Configuration Loading

```python
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("config/config.yaml")
model_config = OmegaConf.load("config/model/gpt2_small.yaml")
loss_config = OmegaConf.load("config/loss/tidpo.yaml")

# Merge configurations
config = OmegaConf.merge(config, model_config, loss_config)
```

## Utilities

### `utils.py`

Utility functions for training.

#### `slice_and_move_batch_for_device()`

```python
def slice_and_move_batch_for_device(batch, rank, world_size, device):
    """
    Slice batch for multi-GPU training and move to device.
    
    Args:
        batch: Input batch
        rank: GPU rank
        world_size: Number of GPUs
        device: Target device
        
    Returns:
        on_device: Batch moved to device
    """
```

#### `get_local_run_dir()`

```python
def get_local_run_dir(exp_name, local_dirs):
    """
    Get local run directory for experiment.
    
    Args:
        exp_name: Experiment name
        local_dirs: List of possible directories
        
    Returns:
        run_dir: Selected run directory
    """
```

## Testing

### Test Modules

#### `test_gradient_attribution.py`

Tests for gradient attribution functionality:

```python
def test_basic_gradient_attribution():
    """Test basic gradient attribution functionality."""
    
def test_edge_cases():
    """Test edge cases for gradient attribution."""
```

#### `test_tidpo.py`

Tests for TIDPO functionality:

```python
def test_gradient_attribution():
    """Test gradient attribution functionality."""
    
def test_weighted_loss():
    """Test weighted loss function."""
    
def test_config_loading():
    """Test configuration loading."""
```

#### `test_triplet_loss.py`

Tests for triplet loss:

```python
def test_triplet_loss_calculation():
    """Test triplet loss calculation."""
    
def test_log_ratio_sequence():
    """Test log-ratio sequence calculation."""
    
def test_numerical_stability():
    """Test numerical stability."""
```

### Running Tests

```bash
# Run all tests
python test_gradient_attribution.py
python test_tidpo.py
python test_triplet_loss.py
python test_batch_size_fix.py

# Debug specific issues
python debug_batch_issue.py
```

## Advanced Usage

### Custom Token Importance

```python
def custom_importance_function(model, tokenizer, text, device):
    """Custom token importance calculation."""
    # Get base importance scores
    tokens, importances = compute_language_model_gradient_attribution(
        model, tokenizer, text, device
    )
    
    # Apply custom logic
    modified_importances = []
    for importance in importances:
        # Example: Apply smoothing
        smoothed = importance * 0.8 + 0.2
        modified_importances.append(smoothed)
    
    return tokens, modified_importances
```

### Custom Loss Function

```python
def custom_tdpo_loss(chosen_logps, rejected_logps, beta=0.1):
    """Custom TIDPO loss implementation."""
    logits = beta * (chosen_logps - rejected_logps)
    losses = -F.logsigmoid(logits)
    return losses
```

### Multi-GPU Training

```python
# Initialize distributed training
torch.distributed.init_process_group(backend='nccl')

# Create trainer with distributed setup
trainer = BasicTrainer(
    policy=model,
    config=config,
    seed=0,
    run_dir=run_dir,
    reference_model=reference_model
)

# Train with distributed data parallel
trainer.train()
```

## Error Handling

### Common Exceptions

#### `GradientAttributionError`

Raised when gradient attribution fails:

```python
try:
    tokens, importances = compute_language_model_gradient_attribution(
        model, tokenizer, text, device
    )
except Exception as e:
    print(f"Gradient attribution failed: {e}")
    # Fallback to uniform weights
    importances = [1.0] * len(tokens)
```

#### `MemoryError`

Handled with memory optimization:

```python
# Enable memory optimizations
config.activation_checkpointing = True
config.batch_size = 2
config.gradient_accumulation_steps = 4
```

### Debug Mode

```python
# Enable debug mode for detailed output
config.debug = True

# Run with debug information
python train.py debug=true
```

## Performance Tips

### Memory Optimization

1. **Use gradient checkpointing:**
   ```yaml
   activation_checkpointing: true
   ```

2. **Reduce batch size:**
   ```yaml
   batch_size: 2
   gradient_accumulation_steps: 4
   ```

3. **Use mixed precision:**
   ```yaml
   policy_dtype: float16
   reference_dtype: float16
   ```

### Computational Optimization

1. **Cache gradient attribution results**
2. **Use batch processing for token importance**
3. **Enable parallel computation**

### Training Stability

1. **Use gradient clipping:**
   ```yaml
   max_grad_norm: 1.0
   ```

2. **Monitor loss values**
3. **Use learning rate scheduling**

## Examples

### Basic Training

```python
from trainers import BasicTrainer
from omegaconf import OmegaConf

# Load configuration
config = OmegaConf.load("config/config.yaml")

# Initialize trainer
trainer = BasicTrainer(
    policy=model,
    config=config,
    seed=0,
    run_dir="./experiments"
)

# Train
trainer.train()
```

### Custom Training Loop

```python
# Custom training loop
for epoch in range(config.n_epochs):
    for batch in train_iterator:
        # Forward pass
        outputs = trainer.tdpo_concatenated_forward(
            trainer.policy, trainer.reference_model, batch
        )
        
        # Compute loss
        loss = tdpo_loss(*outputs[:-1])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

### Evaluation

```python
# Run evaluation
eval_results = trainer.eval()

# Print results
print(f"Evaluation loss: {eval_results['loss']}")
print(f"Preference accuracy: {eval_results['preference_accuracy']}")
```

This API documentation provides comprehensive information about all major components of the TIDPO implementation. For more detailed examples and advanced usage patterns, refer to the test files and example scripts. 