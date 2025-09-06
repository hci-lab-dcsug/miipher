#!/usr/bin/env python3
"""
Setup script for Miipher-2 with real USM and WaveFit implementations.

This script helps set up the environment for Miipher-2 with actual
USM and WaveFit models instead of placeholders.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def run_command(command, description):
    """Run a command and handle errors."""
    logging.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"✗ {description} failed: {e}")
        logging.error(f"Error output: {e.stderr}")
        return False

def install_requirements():
    """Install Python requirements."""
    logging.info("Installing Python requirements...")
    
    requirements_file = Path(__file__).parent / "requirements_miipher2.txt"
    if requirements_file.exists():
        return run_command(
            f"pip install -r {requirements_file}",
            "Installing Miipher-2 requirements"
        )
    else:
        logging.warning("Requirements file not found, installing basic requirements")
        basic_requirements = [
            "torch>=2.1.0",
            "torchaudio>=0.12.0",
            "transformers>=4.53.0",
            "soundfile>=0.12.1",
            "lightning>=2.0.5"
        ]
        
        for req in basic_requirements:
            if not run_command(f"pip install {req}", f"Installing {req}"):
                return False
        
        return True

def setup_wavefit_repository():
    """Setup WaveFit repository."""
    logging.info("Setting up WaveFit repository...")
    
    wavefit_dir = Path(__file__).parent / "wavefit-pytorch"
    
    if not wavefit_dir.exists():
        logging.info("Cloning WaveFit repository...")
        if not run_command(
            "git clone https://github.com/yukara-ikemiya/wavefit-pytorch.git",
            "Cloning WaveFit repository"
        ):
            return False
    
    # Install WaveFit
    if wavefit_dir.exists():
        logging.info("Installing WaveFit...")
        return run_command(
            f"pip install -e {wavefit_dir}",
            "Installing WaveFit"
        )
    
    return True

def create_model_directories():
    """Create directories for model storage."""
    logging.info("Creating model directories...")
    
    directories = [
        "models/usm",
        "models/wavefit",
        "outputs",
        "logs"
    ]
    
    for directory in directories:
        dir_path = Path(__file__).parent / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"✓ Created directory: {directory}")
    
    return True

def download_usm_model():
    """Download USM model (optional)."""
    logging.info("USM model will be downloaded automatically on first use")
    logging.info("Model: Atotti/google-usm")
    logging.info("This may take some time depending on your internet connection")
    return True

def test_installation():
    """Test the installation."""
    logging.info("Testing installation...")
    
    test_script = """
import torch
from miipher.model.usm_integration import USMFeatureExtractor
from miipher.model.wavefit_integration import WaveFitVocoder
from miipher.model.miipher2 import Miipher2Complete

print("✓ All imports successful")

# Test USM
usm = USMFeatureExtractor()
print("✓ USM Feature Extractor created")

# Test WaveFit
wavefit = WaveFitVocoder()
print("✓ WaveFit Vocoder created")

# Test Miipher-2
miipher2 = Miipher2Complete()
print("✓ Miipher-2 Complete model created")

print("✓ All tests passed!")
"""
    
    try:
        exec(test_script)
        return True
    except Exception as e:
        logging.error(f"✗ Test failed: {e}")
        logging.error("Note: USM and WaveFit models require internet connection for first-time download")
        return False

def main():
    """Main setup function."""
    setup_logging()
    
    logging.info("🚀 Setting up Miipher-2 with real USM and WaveFit implementations")
    
    steps = [
        ("Installing requirements", install_requirements),
        ("Setting up WaveFit repository", setup_wavefit_repository),
        ("Creating model directories", create_model_directories),
        ("Preparing USM model", download_usm_model),
        ("Testing installation", test_installation)
    ]
    
    for step_name, step_function in steps:
        logging.info(f"\n📋 {step_name}...")
        if not step_function():
            logging.error(f"❌ Setup failed at: {step_name}")
            sys.exit(1)
    
    logging.info("\n🎉 Miipher-2 setup completed successfully!")
    logging.info("\n📝 Next steps:")
    logging.info("1. Train Miipher-2: python examples/train_miipher2.py --config examples/configs/config_miipher2.yaml")
    logging.info("2. Run inference: python examples/demo_miipher2.py --model_path path/to/checkpoint.ckpt --input_audio audio.wav")
    logging.info("3. Check the README_MIIPHER2.md for detailed usage instructions")

if __name__ == "__main__":
    main()
