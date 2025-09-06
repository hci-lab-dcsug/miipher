import torch
import torch.nn as nn
from typing import Optional, List
import math
from .usm_integration import USMFeatureExtractor
from .wavefit_integration import WaveFitVocoder


class ParallelAdapter(nn.Module):
    """
    Parallel Adapter for Miipher-2.
    Efficiently predicts clean USM features from noisy inputs.
    """
    def __init__(self, usm_dim: int, hidden_dim: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.usm_dim = usm_dim
        self.hidden_dim = hidden_dim
        
        # Efficient adapter with residual connection
        self.adapter = nn.Sequential(
            nn.Linear(usm_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, usm_dim)
        )
        
        # Initialize adapter weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize adapter weights to be small for stable training."""
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.01)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, noisy_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy_features: (batch_size, seq_len, usm_dim)
        Returns:
            clean_features: (batch_size, seq_len, usm_dim)
        """
        # Residual connection with adapter
        adapter_output = self.adapter(noisy_features)
        return noisy_features + adapter_output


class Miipher2(nn.Module):
    """
    Miipher-2: Universal Speech Restoration Model
    
    Key improvements over Miipher:
    1. Uses Universal Speech Model (USM) instead of WavLM
    2. Parallel adapters instead of iterative Conformer blocks
    3. Conditioning-free (no speaker/phone conditioning)
    4. More efficient architecture
    """
    def __init__(
        self,
        usm_dim: int = 1024,
        n_adapters: int = 12,
        adapter_hidden_dim: int = 1024,
        dropout: float = 0.1,
        freeze_usm: bool = True
    ):
        super().__init__()
        self.usm_dim = usm_dim
        self.n_adapters = n_adapters
        self.freeze_usm = freeze_usm
        
        # Parallel adapters for each USM layer
        self.parallel_adapters = nn.ModuleList([
            ParallelAdapter(usm_dim, adapter_hidden_dim, dropout)
            for _ in range(n_adapters)
        ])
        
        # Optional: Layer normalization for stability
        self.layer_norm = nn.LayerNorm(usm_dim)
        
        # Optional: Final projection layer
        self.final_projection = nn.Sequential(
            nn.Linear(usm_dim, usm_dim),
            nn.LayerNorm(usm_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        noisy_usm_features: torch.Tensor,
        feature_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            noisy_usm_features: (batch_size, seq_len, usm_dim) - Noisy USM features
            feature_lengths: (batch_size,) - Optional sequence lengths for masking
        Returns:
            clean_features: (batch_size, seq_len, usm_dim) - Clean USM features
        """
        batch_size, seq_len, _ = noisy_usm_features.shape
        
        # Apply parallel adapters
        clean_features = noisy_usm_features
        for adapter in self.parallel_adapters:
            clean_features = adapter(clean_features)
        
        # Apply layer normalization
        clean_features = self.layer_norm(clean_features)
        
        # Apply final projection
        clean_features = self.final_projection(clean_features)
        
        return clean_features
    
    def get_num_parameters(self) -> dict:
        """Get parameter count for analysis."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "adapter_parameters": sum(p.numel() for p in self.parallel_adapters.parameters()),
            "efficiency_ratio": trainable_params / total_params if total_params > 0 else 0
        }


# USMFeatureExtractor is now imported from usm_integration.py


class Miipher2Complete(nn.Module):
    """
    Complete Miipher-2 system with real USM and WaveFit integration.
    """
    def __init__(
        self,
        usm_model_name: str = "Atotti/google-usm",
        usm_dim: int = 1536,  # Updated for real USM
        n_adapters: int = 12,
        adapter_hidden_dim: int = 1024,
        freeze_usm: bool = True,
        wavefit_model_path: Optional[str] = None,
        wavefit_model_type: str = "wavefit-3_mem-efficient"
    ):
        super().__init__()
        
        # USM feature extractor (real implementation)
        self.usm_extractor = USMFeatureExtractor(
            model_name=usm_model_name,
            freeze=freeze_usm
        )
        
        # Miipher-2 restoration model
        self.miipher2 = Miipher2(
            usm_dim=usm_dim,
            n_adapters=n_adapters,
            adapter_hidden_dim=adapter_hidden_dim,
            freeze_usm=freeze_usm
        )
        
        # WaveFit vocoder (real implementation)
        self.wavefit_vocoder = WaveFitVocoder(
            model_path=wavefit_model_path,
            model_type=wavefit_model_type
        )
    
    def forward(self, noisy_audio: torch.Tensor) -> torch.Tensor:
        """
        Complete forward pass: noisy audio -> clean audio.
        
        Args:
            noisy_audio: (batch_size, audio_length) - Noisy input audio
        Returns:
            clean_audio: (batch_size, audio_length) - Clean output audio
        """
        # Extract USM features from noisy audio
        noisy_usm_features = self.usm_extractor.extract_features(noisy_audio)
        
        # Apply Miipher-2 restoration
        clean_features = self.miipher2(noisy_usm_features)
        
        # Synthesize clean audio with WaveFit
        clean_audio = self.wavefit_vocoder.synthesize(clean_features)
        
        return clean_audio
    
    def get_model_info(self) -> dict:
        """Get comprehensive model information."""
        info = {
            "model_name": "Miipher-2",
            "usm_info": self.usm_extractor.get_model_info(),
            "miipher2_params": self.miipher2.get_num_parameters(),
            "wavefit_info": self.wavefit_vocoder.get_model_info(),
            "total_params": sum(p.numel() for p in self.parameters()),
            "trainable_params": sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        return info

