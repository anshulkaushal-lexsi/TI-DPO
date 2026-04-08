#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TIDPO (Token Importance DPO) training example script
# 
# This script demonstrates how to use TIDPO for training, including:
# 1. Supervised Fine-tuning (SFT) stage
# 2. TIDPO preference learning stage
# 
# Usage:
# python run_tidpo_example.py

import subprocess
import sys
import os

def run_command(command, description):
    """Run command and handle errors"""
    try:
        print(f"Executing: {description}")
        print(f"Command: {command}")
        
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        
        print("✅ Execution successful!")
        return True
        
    except subprocess.CalledProcessError as e:
        print("❌ Execution failed!")
        print(f"Error code: {e.returncode}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Main training function"""
    print("🚀 Starting TIDPO training pipeline")
    print("=" * 50)
    
    # Experiment name
    exp_name = "tidpo_example"
    
    # Step 1: Supervised Fine-tuning (SFT)
    print("\n📚 Step 1: Execute Supervised Fine-tuning (SFT)")
    
    sft_command = f"""python -u train.py \\
        model=gpt2_small \\
        datasets=[hh] \\
        loss=sft \\
        exp_name={exp_name}_sft \\
        batch_size=4 \\
        eval_batch_size=4 \\
        n_epochs=1 \\
        n_eval_examples=8 \\
        eval_every=100 \\
        lr=1e-5 \\
        max_length=256 \\
        max_prompt_length=128 \\
        gradient_accumulation_steps=1 \\
        activation_checkpointing=true \\
        sample_during_eval=false \\
        do_first_eval=false"""
    
    if not run_command(sft_command, "Supervised Fine-tuning"):
        print("❌ SFT training failed, exiting")
        return False
    
    # Step 2: TIDPO Training
    print("\n🎯 Step 2: Execute TIDPO Training")
    
    tidpo_command = f"""python -u train.py \\
        model=gpt2_small \\
        datasets=[hh] \\
        loss=tidpo \\
        exp_name={exp_name}_tidpo \\
        batch_size=4 \\
        eval_batch_size=4 \\
        n_epochs=1 \\
        n_eval_examples=8 \\
        eval_every=100 \\
        lr=1e-5 \\
        max_length=256 \\
        max_prompt_length=128 \\
        gradient_accumulation_steps=1 \\
        activation_checkpointing=true \\
        sample_during_eval=false \\
        do_first_eval=false"""
    
    if not run_command(tidpo_command, "TIDPO Training"):
        print("❌ TIDPO training failed, exiting")
        return False
    
    print("\n🎉 TIDPO training pipeline completed successfully!")
    print("\n📊 Training results:")
    print(f"- SFT model: .cache/{exp_name}_sft_*/")
    print(f"- TIDPO model: .cache/{exp_name}_tidpo_*/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 