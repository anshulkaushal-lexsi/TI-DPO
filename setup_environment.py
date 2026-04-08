#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Environment setup script

Resolve dataset caching and path issues
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

def setup_training_environment():
    """Setup training environment"""
    print("🔧 Setting up TIDPO training environment...")
    
    # 1. Set environment variables
    print("📝 Setting environment variables...")
    os.environ.setdefault('HF_HOME', '.cache/huggingface')
    os.environ.setdefault('TRANSFORMERS_CACHE', '.cache/huggingface/transformers')
    os.environ.setdefault('HF_DATASETS_CACHE', '.cache/huggingface/datasets')
    
    print(f"✅ Environment variables set")
    
    # 2. Create cache directories
    print("📁 Creating cache directories...")
    cache_dirs = [
        '.cache',
        '.cache/huggingface',
        '.cache/huggingface/transformers',
        '.cache/huggingface/datasets',
        '.cache/huggingface/hub',
    ]
    
    for cache_dir in cache_dirs:
        try:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {cache_dir}")
        except Exception as e:
            print(f"⚠️  Cannot create directory {cache_dir}: {e}")
    
    # 3. Clean potentially corrupted cache
    print("🧹 Cleaning corrupted cache...")
    try:
        # Find and delete corrupted cache files
        cache_root = Path('.cache')
        if cache_root.exists():
            for file_path in cache_root.rglob('*.lock'):
                try:
                    file_path.unlink()
                    print(f"🗑️  Deleted lock file: {file_path}")
                except Exception as e:
                    print(f"⚠️  Cannot delete {file_path}: {e}")
    except Exception as e:
        print(f"⚠️  Error cleaning cache: {e}")
    
    print("✅ Environment setup completed!")

def test_dataset_loading():
    """Test dataset loading"""
    print("\n🧪 Testing dataset loading...")
    
    try:
        from datasets import load_dataset
        
        # Set cache directory
        os.environ['HF_DATASETS_CACHE'] = 'C:/tmp/huggingface/datasets'
        
        print("📥 Downloading HH dataset...")
        dataset = load_dataset('Anthropic/hh-rlhf', split='train[:10]')  # Load only the first 10 samples for testing
        
        print(f"✅ Dataset loaded successfully!")
        print(f"Dataset size: {len(dataset)}")
        print(f"Dataset features: {dataset.features}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 TIDPO Environment Setup Tool")
    print("=" * 50)
    
    # Set environment
    setup_training_environment()
    
    # Test dataset loading
    if test_dataset_loading():
        print("\n🎉 Environment setup successful! You can start training.")
        print("\n💡 Use the following command to start training:")
        print("python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=my_tidpo")
        return True
    else:
        print("\n❌ Environment setup failed, please check your network connection and disk space")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 