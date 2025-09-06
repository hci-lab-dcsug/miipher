from typing import Any, Optional, Dict
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from omegaconf import DictConfig
import hydra
import logging

from .model.miipher2 import Miipher2Complete, Miipher2
from .model.usm_integration import USMFeatureExtractor
from .model.wavefit_integration import WaveFitVocoder


class Miipher2LightningModule(LightningModule):
    """
    Lightning module for Miipher-2: Universal Speech Restoration Model.
    
    Key differences from original Miipher:
    1. No speaker/phone conditioning (conditioning-free)
    2. Uses USM instead of WavLM
    3. Parallel adapters instead of iterative Conformer blocks
    4. More efficient training and inference
    """
    
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        
        # Initialize Miipher-2 model
        self.miipher2 = Miipher2Complete(
            usm_model_name=cfg.model.usm_model.pretrained_model_name_or_path,
            usm_dim=cfg.model.miipher2.usm_dim,
            n_adapters=cfg.model.miipher2.n_adapters,
            adapter_hidden_dim=cfg.model.miipher2.adapter_hidden_dim,
            freeze_usm=cfg.model.usm_model.get('freeze', True)
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.smooth_l1_loss = nn.SmoothL1Loss()
        
        # Optional: Add perceptual loss for better quality
        self.use_perceptual_loss = cfg.model.get('use_perceptual_loss', False)
        if self.use_perceptual_loss:
            self.perceptual_loss = self._create_perceptual_loss()
        
        # Training metrics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        
        # Log model info
        self._log_model_info()
    
    def _create_perceptual_loss(self):
        """Create perceptual loss using a pre-trained model."""
        # Placeholder for perceptual loss implementation
        # In practice, this could use a pre-trained speech model for perceptual loss
        return nn.MSELoss()  # Fallback to MSE for now
    
    def _log_model_info(self):
        """Log model architecture information."""
        model_info = self.miipher2.get_model_info()
        logging.info(f"Miipher-2 Model Info: {model_info}")
        
        # Log parameter efficiency
        total_params = model_info['total_params']
        trainable_params = model_info['trainable_params']
        efficiency = trainable_params / total_params if total_params > 0 else 0
        
        logging.info(f"Parameter Efficiency: {efficiency:.4f} ({trainable_params:,}/{total_params:,} trainable)")
    
    def forward(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for inference.
        
        Args:
            noisy_audio: (batch_size, audio_length) - Noisy input audio
        Returns:
            clean_features: (batch_size, seq_len, usm_dim) - Clean USM features
        """
        return self.miipher2(noisy_audio)
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """
        Training step for Miipher-2.
        
        Args:
            batch: Dictionary containing:
                - 'noisy_audio': (batch_size, audio_length) - Noisy audio
                - 'clean_audio': (batch_size, audio_length) - Clean reference audio
                - 'audio_lengths': (batch_size,) - Audio sequence lengths
        """
        noisy_audio = batch['noisy_audio']
        clean_audio = batch['clean_audio']
        audio_lengths = batch.get('audio_lengths', None)
        
        # Forward pass through Miipher-2
        clean_features = self.miipher2(noisy_audio)
        
        # Extract clean USM features for target
        with torch.no_grad():
            clean_target_features = self.miipher2.usm_extractor(clean_audio)
        
        # Compute loss
        loss = self._compute_loss(clean_features, clean_target_features, audio_lengths)
        
        # Log training metrics
        self.log('train/loss', loss, batch_size=noisy_audio.size(0), prog_bar=True)
        self.log('train/mse_loss', self.mse_loss(clean_features, clean_target_features), 
                batch_size=noisy_audio.size(0))
        self.log('train/mae_loss', self.mae_loss(clean_features, clean_target_features), 
                batch_size=noisy_audio.size(0))
        
        # Store outputs for epoch-end logging
        self.training_step_outputs.append({
            'loss': loss,
            'batch_size': noisy_audio.size(0)
        })
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> STEP_OUTPUT:
        """
        Validation step for Miipher-2.
        """
        noisy_audio = batch['noisy_audio']
        clean_audio = batch['clean_audio']
        audio_lengths = batch.get('audio_lengths', None)
        
        # Forward pass
        clean_features = self.miipher2(noisy_audio)
        
        # Extract clean USM features for target
        with torch.no_grad():
            clean_target_features = self.miipher2.usm_extractor(clean_audio)
        
        # Compute loss
        loss = self._compute_loss(clean_features, clean_target_features, audio_lengths)
        
        # Log validation metrics
        self.log('val/loss', loss, batch_size=noisy_audio.size(0), prog_bar=True)
        self.log('val/mse_loss', self.mse_loss(clean_features, clean_target_features), 
                batch_size=noisy_audio.size(0))
        self.log('val/mae_loss', self.mae_loss(clean_features, clean_target_features), 
                batch_size=noisy_audio.size(0))
        
        # Store outputs for epoch-end logging
        self.validation_step_outputs.append({
            'loss': loss,
            'batch_size': noisy_audio.size(0)
        })
        
        return loss
    
    def _compute_loss(
        self, 
        predicted_features: torch.Tensor, 
        target_features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute combined loss for Miipher-2 training.
        
        Args:
            predicted_features: (batch_size, seq_len, usm_dim) - Predicted clean features
            target_features: (batch_size, seq_len, usm_dim) - Target clean features
            lengths: (batch_size,) - Optional sequence lengths for masking
        """
        # Primary MSE loss
        mse_loss = self.mse_loss(predicted_features, target_features)
        
        # Additional MAE loss for robustness
        mae_loss = self.mae_loss(predicted_features, target_features)
        
        # Smooth L1 loss for better gradient flow
        smooth_l1_loss = self.smooth_l1_loss(predicted_features, target_features)
        
        # Combine losses
        total_loss = mse_loss + 0.1 * mae_loss + 0.1 * smooth_l1_loss
        
        # Optional perceptual loss
        if self.use_perceptual_loss:
            perceptual_loss = self.perceptual_loss(predicted_features, target_features)
            total_loss += 0.1 * perceptual_loss
        
        return total_loss
    
    def on_train_epoch_end(self) -> None:
        """Log epoch-end training metrics."""
        if not self.training_step_outputs:
            return
        
        # Compute epoch averages
        total_loss = sum(output['loss'] * output['batch_size'] for output in self.training_step_outputs)
        total_samples = sum(output['batch_size'] for output in self.training_step_outputs)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        self.log('train/epoch_loss', avg_loss, on_epoch=True)
        
        # Clear outputs
        self.training_step_outputs.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Log epoch-end validation metrics."""
        if not self.validation_step_outputs:
            return
        
        # Compute epoch averages
        total_loss = sum(output['loss'] * output['batch_size'] for output in self.validation_step_outputs)
        total_samples = sum(output['batch_size'] for output in self.validation_step_outputs)
        avg_loss = total_loss / total_samples if total_samples > 0 else 0
        
        self.log('val/epoch_loss', avg_loss, on_epoch=True)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizers for Miipher-2 training."""
        # Only optimize Miipher-2 parameters (USM should be frozen)
        miipher2_params = list(self.miipher2.miipher2.parameters())
        
        optimizer = hydra.utils.instantiate(
            self.cfg.optimizers,
            params=miipher2_params
        )
        
        # Optional learning rate scheduler
        if 'scheduler' in self.cfg:
            scheduler = hydra.utils.instantiate(
                self.cfg.scheduler,
                optimizer=optimizer
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss'
                }
            }
        
        return optimizer
    
    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Prediction step for inference.
        
        Args:
            batch: Dictionary containing 'noisy_audio'
        Returns:
            clean_features: Clean USM features
        """
        noisy_audio = batch['noisy_audio']
        return self.miipher2(noisy_audio)
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        model_info = self.miipher2.get_model_info()
        
        summary = {
            'model_name': 'Miipher-2',
            'architecture': 'Universal Speech Restoration with Parallel Adapters',
            'conditioning': 'None (conditioning-free)',
            'usm_model': model_info['usm_model'],
            'usm_frozen': model_info['usm_frozen'],
            'total_parameters': model_info['total_params'],
            'trainable_parameters': model_info['trainable_params'],
            'parameter_efficiency': model_info['trainable_params'] / model_info['total_params'],
            'adapters': {
                'count': self.cfg.model.miipher2.n_adapters,
                'hidden_dim': self.cfg.model.miipher2.adapter_hidden_dim,
                'parameters': model_info['miipher2_params']['adapter_parameters']
            }
        }
        
        return summary


class Miipher2Inference:
    """
    Inference wrapper for Miipher-2 model.
    Provides easy-to-use interface for speech restoration.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'cuda'):
        """
        Initialize Miipher-2 inference.
        
        Args:
            checkpoint_path: Path to trained Miipher-2 checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device)
        
        # Load model from checkpoint
        self.model = Miipher2LightningModule.load_from_checkpoint(
            checkpoint_path,
            map_location=self.device
        )
        self.model.eval()
        self.model.to(self.device)
        
        logging.info("Miipher-2 model loaded for inference")
    
    def restore_speech(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """
        Restore speech from noisy audio.
        
        Args:
            noisy_audio: (audio_length,) or (batch_size, audio_length) - Noisy audio
        Returns:
            clean_features: Clean USM features
        """
        with torch.no_grad():
            if noisy_audio.dim() == 1:
                noisy_audio = noisy_audio.unsqueeze(0)  # Add batch dimension
            
            noisy_audio = noisy_audio.to(self.device)
            clean_features = self.model(noisy_audio)
            
            return clean_features.cpu()
    
    def batch_restore(self, noisy_audio_batch: torch.Tensor) -> torch.Tensor:
        """
        Restore speech for a batch of noisy audio.
        
        Args:
            noisy_audio_batch: (batch_size, audio_length) - Batch of noisy audio
        Returns:
            clean_features_batch: (batch_size, seq_len, usm_dim) - Clean features
        """
        return self.restore_speech(noisy_audio_batch)

