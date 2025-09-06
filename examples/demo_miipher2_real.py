#!/usr/bin/env python3
"""
Demo script for Miipher-2 with real USM and WaveFit implementations.

This script demonstrates how to use Miipher-2 with actual USM and WaveFit models
for high-quality speech restoration.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torchaudio
import numpy as np
from omegaconf import DictConfig, OmegaConf
import hydra

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from miipher.model.miipher2 import Miipher2Complete
from miipher.model.usm_integration import USMFeatureExtractor
from miipher.model.wavefit_integration import WaveFitVocoder


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def load_audio(file_path: str, target_sr: int = 22050) -> torch.Tensor:
    """
    Load audio file and resample to target sample rate.
    
    Args:
        file_path: Path to audio file
        target_sr: Target sample rate
        
    Returns:
        audio: Audio tensor (samples,)
    """
    # Load audio
    audio, sr = torchaudio.load(file_path)
    
    # Convert to mono if stereo
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    
    # Resample if necessary
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        audio = resampler(audio)
    
    # Convert to 1D tensor
    audio = audio.squeeze(0)
    
    return audio


def save_audio(audio: torch.Tensor, file_path: str, sample_rate: int = 22050) -> None:
    """
    Save audio tensor to file.
    
    Args:
        audio: Audio tensor (samples,)
        file_path: Output file path
        sample_rate: Sample rate
    """
    # Ensure audio is 2D for torchaudio
    if audio.dim() == 1:
        audio = audio.unsqueeze(0)
    
    # Save audio
    torchaudio.save(file_path, audio, sample_rate)


def test_usm_extraction(audio: torch.Tensor, device: str = "cuda") -> None:
    """Test USM feature extraction."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing USM feature extraction...")
    
    # Create USM extractor
    usm_extractor = USMFeatureExtractor(
        model_name="Atotti/google-usm",
        freeze=True,
        device=device
    )
    
    # Extract features
    features = usm_extractor.extract_features(audio.unsqueeze(0))
    
    logger.info(f"USM features shape: {features.shape}")
    logger.info(f"USM model info: {usm_extractor.get_model_info()}")


def test_wavefit_synthesis(features: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Test WaveFit audio synthesis."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing WaveFit audio synthesis...")
    
    # Create WaveFit vocoder
    wavefit_vocoder = WaveFitVocoder(
        model_path=None,  # Initialize without pretrained weights
        model_type="wavefit-3_mem-efficient",
        device=device
    )
    
    # Synthesize audio
    audio = wavefit_vocoder.synthesize(features)
    
    logger.info(f"Generated audio shape: {audio.shape}")
    logger.info(f"WaveFit model info: {wavefit_vocoder.get_model_info()}")
    
    return audio


def test_complete_miipher2(noisy_audio: torch.Tensor, device: str = "cuda") -> torch.Tensor:
    """Test complete Miipher-2 pipeline."""
    logger = logging.getLogger(__name__)
    
    logger.info("Testing complete Miipher-2 pipeline...")
    
    # Create complete Miipher-2 model
    miipher2 = Miipher2Complete(
        usm_model_name="Atotti/google-usm",
        usm_dim=1536,
        n_adapters=12,
        adapter_hidden_dim=1024,
        freeze_usm=True,
        wavefit_model_path=None,  # Initialize without pretrained weights
        wavefit_model_type="wavefit-3_mem-efficient"
    )
    
    # Move to device
    miipher2 = miipher2.to(device)
    miipher2.eval()
    
    # Process audio
    with torch.no_grad():
        clean_audio = miipher2(noisy_audio.unsqueeze(0))
    
    logger.info(f"Input audio shape: {noisy_audio.shape}")
    logger.info(f"Output audio shape: {clean_audio.shape}")
    logger.info(f"Model info: {miipher2.get_model_info()}")
    
    return clean_audio.squeeze(0)


def process_single_audio(
    input_path: str,
    output_dir: str,
    device: str = "cuda",
    test_components: bool = True
) -> None:
    """
    Process a single audio file with Miipher-2.
    
    Args:
        input_path: Path to input audio file
        output_dir: Output directory
        device: Device to run on
        test_components: Whether to test individual components
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input audio
    logger.info(f"Loading audio from {input_path}")
    noisy_audio = load_audio(input_path)
    logger.info(f"Audio shape: {noisy_audio.shape}, Duration: {len(noisy_audio) / 22050:.2f}s")
    
    # Move to device
    noisy_audio = noisy_audio.to(device)
    
    # Test individual components if requested
    if test_components:
        test_usm_extraction(noisy_audio, device)
        
        # Test WaveFit with dummy features
        dummy_features = torch.randn(1, 100, 1536).to(device)
        test_wavefit_synthesis(dummy_features, device)
    
    # Process with complete Miipher-2
    logger.info("Processing with complete Miipher-2...")
    clean_audio = test_complete_miipher2(noisy_audio, device)
    
    # Save results
    output_path = os.path.join(output_dir, "restored_audio.wav")
    save_audio(clean_audio.cpu(), output_path)
    logger.info(f"Saved restored audio to {output_path}")
    
    # Save input for comparison
    input_output_path = os.path.join(output_dir, "input_audio.wav")
    save_audio(noisy_audio.cpu(), input_output_path)
    logger.info(f"Saved input audio to {input_output_path}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Miipher-2 Real Implementation Demo")
    parser.add_argument(
        "--input_audio",
        type=str,
        required=True,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./miipher2_outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on"
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--test_components",
        action="store_true",
        help="Test individual components (USM, WaveFit)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not os.path.exists(args.input_audio):
        logger.error(f"Input audio file not found: {args.input_audio}")
        return
    
    # Process audio
    try:
        process_single_audio(
            input_path=args.input_audio,
            output_dir=args.output_dir,
            device=args.device,
            test_components=args.test_components
        )
        
        logger.info("🎉 Processing completed successfully!")
        logger.info(f"Check the output directory: {args.output_dir}")
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        raise


if __name__ == "__main__":
    main()
