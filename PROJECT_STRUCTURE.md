# Project Structure

This document provides a detailed overview of the Token Importance Direct Preference Optimization (TIDPO) project structure.

## 📁 Directory Structure

```
Token-level-Direct-Preference-Optimization/
├── 📄 README.md                    # Main project documentation
├── 📄 QUICKSTART.md                # Quick start guide
├── 📄 API_DOCUMENTATION.md         # Detailed API documentation
├── 📄 TROUBLESHOOTING.md           # Troubleshooting guide
├── 📄 PROJECT_STRUCTURE.md         # This file
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
│
├── 🔧 Core Implementation
│   ├── 📄 train.py                 # Main training script
│   ├── 📄 trainers.py              # Training logic and TIDPO implementation
│   ├── 📄 gradient_attribution.py  # Gradient attribution for token importance
│   ├── 📄 preference_datasets.py   # Dataset loading and processing
│   └── 📄 utils.py                 # Utility functions
│
├── ⚙️ Configuration
│   ├── 📄 config.yaml              # Main configuration
│   ├── 📄 config_memory_optimized.yaml  # Memory-optimized settings
│   ├── 📄 config_simple_test.yaml  # Simple test configuration
│   │
│   ├── 📁 loss/                    # Loss function configurations
│   │   ├── 📄 sft.yaml            # Supervised fine-tuning
│   │   └── 📄 tidpo.yaml          # TIDPO configuration
│   │
│   └── 📁 model/                   # Model configurations
│       ├── 📄 blank_model.yaml     # Blank model template
│       ├── 📄 gpt2_small.yaml      # GPT-2 small model
│       ├── 📄 gpt2_large.yaml      # GPT-2 large model
│       ├── 📄 gpt2_xl.yaml         # GPT-2 XL model
│       ├── 📄 gptj.yaml            # GPT-J model
│       ├── 📄 llama7b.yaml         # LLaMA-7B model
│       ├── 📄 pythia28.yaml        # Pythia-2.8B model
│       └── 📄 pythia69.yaml        # Pythia-6.9B model
│
├── 🧪 Testing and Debugging
│   ├── 📄 test_gradient_attribution.py    # Test gradient attribution
│   ├── 📄 test_tidpo.py                   # Test TIDPO functionality
│   ├── 📄 test_triplet_loss.py            # Test triplet loss
│   ├── 📄 test_batch_size_fix.py          # Test batch processing
│   ├── 📄 test_tidpo_fix.py               # Test TIDPO fixes
│   ├── 📄 test_triplet_loss_fix.py        # Test triplet loss fixes
│   └── 📄 debug_batch_issue.py            # Debug batch issues
│
├── 🚀 Examples and Scripts
│   ├── 📄 run_tidpo_example.py            # Complete training example
│   └── 📄 setup_environment.py             # Environment setup script
│
├── 📊 Outputs and Results
│   └── 📁 outputs/                        # Training outputs and logs
│       ├── 📁 2025-07-30/                 # Date-based experiment folders
│       └── 📁 2025-07-31/
│
├── 🗂️ Model Files
│   └── 📁 pythia-2.8b/                    # Pre-trained model files
│       ├── 📄 config.json
│       ├── 📄 generation_config.json
│       ├── 📄 model-00001-of-00003.safetensors
│       ├── 📄 model-00002-of-00003.safetensors
│       ├── 📄 model-00003-of-00003.safetensors
│       ├── 📄 model.safetensors.index.json
│       ├── 📄 read.py
│       ├── 📄 special_tokens_map.json
│       ├── 📄 tokenizer_config.json
│       └── 📄 tokenizer.json
```

## 📋 File Descriptions

### Core Implementation Files

#### `train.py`
- **Purpose**: Main entry point for training
- **Key Functions**: 
  - Parse command line arguments
  - Load configuration
  - Initialize trainer
  - Start training process
- **Usage**: `python train.py model=gpt2_small datasets=[hh] loss=tidpo`

#### `trainers.py`
- **Purpose**: Core training logic and TIDPO implementation
- **Key Classes**:
  - `BasicTrainer`: Main trainer class
- **Key Methods**:
  - `tdpo_concatenated_forward()`: TIDPO forward pass
  - `_compute_token_importance_weights()`: Token importance calculation
  - `_compute_triplet_loss()`: Triplet loss computation
  - `_get_log_ratio_sequence()`: Log-ratio sequence calculation
- **Size**: ~58KB, 1126 lines

#### `gradient_attribution.py`
- **Purpose**: Gradient-based token importance calculation
- **Key Functions**:
  - `compute_language_model_gradient_attribution()`: Main gradient attribution function
  - `compute_gradient_attribution()`: Alternative implementation
- **Features**:
  - L1/L2 norm calculation
  - Error handling and fallbacks
  - Support for multiple model architectures
- **Size**: ~11KB, 304 lines

#### `preference_datasets.py`
- **Purpose**: Dataset loading and processing
- **Key Functions**:
  - `get_batch_iterator()`: Create batch iterators
  - `tokenize_batch_element()`: Tokenize individual samples
  - `get_collate_fn()`: Collate function for batching
- **Supported Datasets**:
  - Anthropic/hh-rlhf (Helpful-Harmful)
  - Stanford Human Preferences (SHP)
- **Size**: ~19KB, 436 lines

#### `utils.py`
- **Purpose**: Utility functions for training
- **Key Functions**:
  - `slice_and_move_batch_for_device()`: Multi-GPU batch handling
  - `get_local_run_dir()`: Directory management
  - `get_model_and_tokenizer()`: Model loading utilities
- **Size**: ~7KB, 191 lines

### Configuration Files

#### `config/config.yaml`
- **Purpose**: Main configuration file
- **Key Parameters**:
  - Training parameters (batch_size, lr, epochs)
  - Model and dataset selection
  - Optimization settings
  - Evaluation parameters

#### `config/loss/tidpo.yaml`
- **Purpose**: TIDPO-specific configuration
- **Key Parameters**:
  - `use_tidpo`: Enable TIDPO
  - `alpha_triplet`: Triplet loss weight
  - `gamma`: Loss combination weight
  - `enable_gradient_attribution`: Enable gradient attribution

#### `config/model/gpt2_small.yaml`
- **Purpose**: GPT-2 small model configuration
- **Key Parameters**:
  - `name_or_path`: Model identifier
  - `policy_dtype`: Policy model data type
  - `reference_dtype`: Reference model data type

### Testing Files

#### `test_gradient_attribution.py`
- **Purpose**: Test gradient attribution functionality
- **Tests**:
  - Basic gradient attribution
  - Edge cases (empty text, single word)
  - Error handling

#### `test_tidpo.py`
- **Purpose**: Test TIDPO functionality
- **Tests**:
  - Gradient attribution
  - Weighted loss calculation
  - Configuration loading

#### `test_triplet_loss.py`
- **Purpose**: Test triplet loss implementation
- **Tests**:
  - Triplet loss calculation
  - Log-ratio sequence computation
  - Numerical stability

#### `debug_batch_issue.py`
- **Purpose**: Debug batch processing issues
- **Features**:
  - Data iterator debugging
  - Collate function testing
  - Tokenization debugging

### Example and Script Files

#### `run_tidpo_example.py`
- **Purpose**: Complete training example
- **Features**:
  - SFT training
  - TIDPO training
  - Automatic configuration
  - Error handling

#### `setup_environment.py`
- **Purpose**: Environment setup and configuration
- **Features**:
  - Cache directory creation
  - Environment variable setup
  - Dataset loading test

## 🔧 Key Components

### Training Pipeline

1. **Data Loading**: `preference_datasets.py`
2. **Model Initialization**: `utils.py`
3. **Forward Pass**: `trainers.py`
4. **Loss Calculation**: `trainers.py`
5. **Optimization**: `trainers.py`
6. **Evaluation**: `trainers.py`

### Gradient Attribution Pipeline

1. **Text Tokenization**: `gradient_attribution.py`
2. **Embedding Computation**: `gradient_attribution.py`
3. **Gradient Calculation**: `gradient_attribution.py`
4. **Importance Scoring**: `gradient_attribution.py`
5. **Weight Application**: `trainers.py`

### Configuration System

1. **Main Config**: `config/config.yaml`
2. **Model Config**: `config/model/*.yaml`
3. **Loss Config**: `config/loss/*.yaml`
4. **Merging**: OmegaConf-based configuration merging

## 📊 File Statistics

| File Type | Count | Total Size | Description |
|-----------|-------|------------|-------------|
| Python Files | 15 | ~150KB | Core implementation and tests |
| Configuration Files | 12 | ~5KB | YAML configuration files |
| Documentation Files | 5 | ~50KB | Markdown documentation |
| Model Files | 10 | ~6GB | Pre-trained model weights |
| Output Files | Variable | Variable | Training logs and results |

## 🎯 Usage Patterns

### For Beginners
1. Start with `QUICKSTART.md`
2. Use `run_tidpo_example.py`
3. Check `config/config.yaml` for settings

### For Researchers
1. Study `trainers.py` for algorithm implementation
2. Examine `gradient_attribution.py` for token importance
3. Modify `config/loss/tidpo.yaml` for experiments

### For Developers
1. Add tests to `test_*.py` files
2. Extend `preference_datasets.py` for new datasets
3. Modify `trainers.py` for new algorithms

## 🔍 File Dependencies

### Core Dependencies
```
train.py
├── trainers.py
├── preference_datasets.py
├── utils.py
└── gradient_attribution.py
```

### Configuration Dependencies
```
config/config.yaml
├── config/loss/tidpo.yaml
├── config/model/gpt2_small.yaml
└── config/config_memory_optimized.yaml
```

### Testing Dependencies
```
test_*.py
├── trainers.py
├── gradient_attribution.py
└── preference_datasets.py
```

## 🚀 Getting Started

1. **Install**: `pip install -r requirements.txt`
2. **Setup**: `python setup_environment.py`
3. **Test**: `python test_gradient_attribution.py`
4. **Run**: `python run_tidpo_example.py`

## 📝 Contributing

When contributing to this project:

1. **Add tests**: Create corresponding test files
2. **Update docs**: Modify relevant documentation
3. **Follow structure**: Maintain the existing organization
4. **Check configs**: Update configuration files if needed

This project structure provides a comprehensive and well-organized implementation of TIDPO with extensive testing, documentation, and examples. 