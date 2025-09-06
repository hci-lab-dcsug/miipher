#!/usr/bin/env python3
"""
Demo script for Miipher-2: Universal Speech Restoration Model.

This script demonstrates how to use Miipher-2 for speech restoration:
- Load a trained Miipher-2 model
- Process noisy audio files
- Save restored features or audio

Usage:
    python demo_miipher2.py --model_path path/to/checkpoint.ckpt --input_audio noisy.wav --output_dir ./outputs
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

from miipher.lightning_module_miipher2 import Miipher2Inference
from miipher.model.miipher2 import Miipher2Complete


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


def process_single_audio(
    model: Miipher2Inference,
    input_path: str,
    output_dir: str,
    save_features: bool = True,
    save_audio: bool = False
) -> None:
    """
    Process a single audio file with Miipher-2.
    
    Args:
        model: Miipher-2 inference model
        input_path: Path to input audio file
        output_dir: Output directory
        save_features: Whether to save USM features
        save_audio: Whether to save reconstructed audio (requires vocoder)
    """
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load input audio
    logger.info(f"Loading audio from {input_path}")
    noisy_audio = load_audio(input_path)
    logger.info(f"Audio shape: {noisy_audio.shape}, Duration: {len(noisy_audio) / 22050:.2f}s")
    
    # Process with Miipher-2
    logger.info("Processing with Miipher-2...")
    clean_features = model.restore_speech(noisy_audio)
    logger.info(f"Restored features shape: {clean_features.shape}")
    
    # Save features
    if save_features:
        features_path = os.path.join(output_dir, "restored_features.pt")
        torch.save(clean_features, features_path)
        logger.info(f"Saved features to {features_path}")
    
    # Save audio (if vocoder is available)
    if save_audio:
        # Note: This requires a vocoder to convert features back to audio
        # For now, we'll just save the input audio as a placeholder
        output_audio_path = os.path.join(output_dir, "restored_audio.wav")
        save_audio(noisy_audio, output_audio_path)  # Placeholder
        logger.warning("Audio reconstruction not implemented - requires vocoder")
        logger.info(f"Saved placeholder audio to {output_audio_path}")


def process_batch_audio(
    model: Miipher2Inference,
    input_dir: str,
    output_dir: str,
    file_extensions: list = ['.wav', '.mp3', '.flac'],
    save_features: bool = True,
    save_audio: bool = False
) -> None:
    """
    Process a batch of audio files with Miipher-2.
    
    Args:
        model: Miipher-2 inference model
        input_dir: Directory containing input audio files
        output_dir: Output directory
        file_extensions: List of audio file extensions to process
        save_features: Whether to save USM features
        save_audio: Whether to save reconstructed audio
    """
    logger = logging.getLogger(__name__)
    
    # Find audio files
    input_path = Path(input_dir)
    audio_files = []
    for ext in file_extensions:
        audio_files.extend(input_path.glob(f"*{ext}"))
        audio_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    if not audio_files:
        logger.warning(f"No audio files found in {input_dir}")
        return
    
    logger.info(f"Found {len(audio_files)} audio files to process")
    
    # Process each file
    for i, audio_file in enumerate(audio_files):
        logger.info(f"Processing {i+1}/{len(audio_files)}: {audio_file.name}")
        
        # Create output subdirectory for this file
        file_output_dir = os.path.join(output_dir, audio_file.stem)
        
        try:
            process_single_audio(
                model=model,
                input_path=str(audio_file),
                output_dir=file_output_dir,
                save_features=save_features,
                save_audio=save_audio
            )
        except Exception as e:
            logger.error(f"Error processing {audio_file.name}: {e}")
            continue


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Miipher-2 Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained Miipher-2 checkpoint"
    )
    parser.add_argument(
        "--input_audio",
        type=str,
        help="Path to input audio file"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        help="Directory containing input audio files"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./miipher2_outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--save_features",
        action="store_true",
        default=True,
        help="Save restored USM features"
    )
    parser.add_argument(
        "--save_audio",
        action="store_true",
        help="Save reconstructed audio (requires vocoder)"
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Validate arguments
    if not args.input_audio and not args.input_dir:
        logger.error("Either --input_audio or --input_dir must be specified")
        return
    
    if args.input_audio and args.input_dir:
        logger.error("Cannot specify both --input_audio and --input_dir")
        return
    
    # Check model path
    if not os.path.exists(args.model_path):
        logger.error(f"Model checkpoint not found: {args.model_path}")
        return
    
    # Load model
    logger.info(f"Loading Miipher-2 model from {args.model_path}")
    try:
        model = Miipher2Inference(args.model_path, device=args.device)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return
    
    # Process audio
    if args.input_audio:
        # Single file processing
        if not os.path.exists(args.input_audio):
            logger.error(f"Input audio file not found: {args.input_audio}")
            return
        
        process_single_audio(
            model=model,
            input_path=args.input_audio,
            output_dir=args.output_dir,
            save_features=args.save_features,
            save_audio=args.save_audio
        )
    
    else:
        # Batch processing
        if not os.path.exists(args.input_dir):
            logger.error(f"Input directory not found: {args.input_dir}")
            return
        
        process_batch_audio(
            model=model,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            save_features=args.save_features,
            save_audio=args.save_audio
        )
    
    logger.info("Processing completed!")


if __name__ == "__main__":
    main()
