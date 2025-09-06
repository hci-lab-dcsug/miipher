#!/usr/bin/env python3
"""
Training script for Miipher-2: Universal Speech Restoration Model.

This script demonstrates how to train Miipher-2 with the new architecture:
- Universal Speech Model (USM) for feature extraction
- Parallel adapters for efficient restoration
- Conditioning-free training (no speaker/phone conditioning)
- Multilingual support

Usage:
    python train_miipher2.py --config configs/config_miipher2.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from miipher.lightning_module_miipher2 import Miipher2LightningModule
from miipher.dataset.datamodule_miipher2 import Miipher2DataModule


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('miipher2_training.log')
        ]
    )


def create_callbacks(cfg: DictConfig) -> list:
    """Create Lightning callbacks for training."""
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.train.callbacks.checkpoint.monitor,
        mode=cfg.train.callbacks.checkpoint.mode,
        save_top_k=cfg.train.callbacks.checkpoint.save_top_k,
        filename=cfg.train.callbacks.checkpoint.filename,
        save_last=True,
        every_n_epochs=cfg.train.callbacks.checkpoint.every_n_epochs
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stopping = EarlyStopping(
        monitor=cfg.train.callbacks.early_stopping.monitor,
        patience=cfg.train.callbacks.early_stopping.patience,
        mode=cfg.train.callbacks.early_stopping.mode,
        verbose=True
    )
    callbacks.append(early_stopping)
    
    # Learning rate monitoring
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    callbacks.append(lr_monitor)
    
    return callbacks


def create_logger(cfg: DictConfig) -> WandbLogger:
    """Create Weights & Biases logger."""
    return WandbLogger(
        project=cfg.train.loggers.wandb.project,
        name=cfg.train.loggers.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_dir=cfg.train.loggers.wandb.get('save_dir', './logs')
    )


def train_miipher2(cfg: DictConfig) -> None:
    """Train Miipher-2 model."""
    
    # Setup logging
    setup_logging(cfg.logging.level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Miipher-2 training")
    logger.info(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Set random seeds for reproducibility
    if cfg.get('seed'):
        L.seed_everything(cfg.seed, workers=True)
        logger.info(f"Set random seed to {cfg.seed}")
    
    # Create data module
    logger.info("Creating data module...")
    data_module = Miipher2DataModule(
        train_dataset_path=cfg.data.train_dataset_path,
        val_dataset_path=cfg.data.val_dataset_path,
        train_batch_size=cfg.data.train_batch_size,
        val_batch_size=cfg.data.val_batch_size,
        num_workers=cfg.data.get('num_workers', 4),
        sample_rate=cfg.data.audio_processor.sr,
        max_audio_length=cfg.data.audio_processor.max_audio_length,
        min_audio_length=cfg.data.audio_processor.min_audio_length,
        pin_memory=cfg.data.get('pin_memory', True),
        persistent_workers=cfg.data.get('persistent_workers', True)
    )
    
    # Create model
    logger.info("Creating Miipher-2 model...")
    model = Miipher2LightningModule(cfg)
    
    # Log model summary
    model_summary = model.get_model_summary()
    logger.info(f"Model Summary: {model_summary}")
    
    # Create callbacks
    callbacks = create_callbacks(cfg)
    
    # Create logger
    wandb_logger = create_logger(cfg)
    
    # Create trainer
    logger.info("Creating trainer...")
    trainer = L.Trainer(
        accelerator=cfg.train.trainer.accelerator,
        devices=cfg.train.trainer.devices,
        max_epochs=cfg.train.trainer.max_epochs,
        check_val_every_n_epoch=cfg.train.trainer.check_val_every_n_epoch,
        gradient_clip_val=cfg.train.trainer.get('gradient_clip_val', 1.0),
        accumulate_grad_batches=cfg.train.trainer.get('accumulate_grad_batches', 1),
        precision=cfg.train.trainer.get('precision', '16-mixed'),
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=cfg.get('deterministic', True),
        log_every_n_steps=cfg.logging.log_every_n_steps
    )
    
    # Start training
    logger.info("Starting training...")
    trainer.fit(model, data_module)
    
    # Test the model
    logger.info("Testing model...")
    trainer.test(model, data_module)
    
    logger.info("Training completed successfully!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train Miipher-2 model")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/config_miipher2.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        help="Override configuration values",
        default=[]
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent / config_path
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Initialize Hydra
    with hydra.initialize(config_path=str(config_path.parent), version_base=None):
        cfg = hydra.compose(config_name=config_path.stem, overrides=args.overrides)
    
    # Train the model
    train_miipher2(cfg)


if __name__ == "__main__":
    main()
