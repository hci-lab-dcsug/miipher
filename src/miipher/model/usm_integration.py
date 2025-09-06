"""
Universal Speech Model (USM) Integration for Miipher-2.

This module provides proper integration with Google's Universal Speech Model
for multilingual speech feature extraction.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging
import soundfile as sf
from transformers import Gemma3nAudioEncoder, Gemma3nAudioFeatureExtractor

class USMFeatureExtractor(nn.Module):
    """
    Real USM Feature Extractor using Atotti/Google-USM model.
    
    This implementation uses the actual USM model from Hugging Face:
    https://huggingface.co/Atotti/Google-USM
    """
    
    def __init__(
        self,
        model_name: str = "Atotti/google-usm",
        source_model_id: str = "google/gemma-3n-e2b-it",
        freeze: bool = True,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        super().__init__()
        self.model_name = model_name
        self.source_model_id = source_model_id
        self.freeze = freeze
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_dir = cache_dir
        
        # USM model components
        self.audio_encoder = None
        self.feature_extractor = None
        
        # Load the models
        self._load_usm_models()
        
        logging.info(f"USM Feature Extractor initialized with {model_name}")
        logging.info(f"Model frozen: {freeze}")
    
    def _load_usm_models(self):
        """Load the actual USM models from Hugging Face."""
        # Load the audio encoder (USM)
        self.audio_encoder = Gemma3nAudioEncoder.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        
        # Load the feature extractor
        self.feature_extractor = Gemma3nAudioFeatureExtractor.from_pretrained(
            self.source_model_id,
            cache_dir=self.cache_dir
        )
        
        # Move to device
        self.audio_encoder = self.audio_encoder.to(self.device)
        
        # Freeze parameters if requested
        if self.freeze:
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
        
        logging.info("USM models loaded successfully")
    
    
    def extract_features(
        self, 
        audio: torch.Tensor, 
        sample_rate: int = 16000,
        return_hidden_states: bool = True
    ) -> torch.Tensor:
        """
        Extract USM features from audio using the real USM model.
        
        Args:
            audio: Input audio tensor (batch_size, audio_length) or (audio_length,)
            sample_rate: Sample rate of input audio
            return_hidden_states: Whether to return hidden states
            
        Returns:
            usm_features: USM features (batch_size, seq_len, feature_dim)
        """
        if self.audio_encoder is None:
            raise RuntimeError("USM model not loaded")
        
        # Ensure audio is 2D (batch_size, audio_length)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Extract features with real USM model
        return self._extract_features_with_usm(audio, sample_rate)
    
    def _extract_features_with_usm(
        self, 
        audio: torch.Tensor, 
        sample_rate: int
    ) -> torch.Tensor:
        """Extract features using the real USM model."""
        # Convert audio to numpy for feature extractor
        audio_np = audio.cpu().numpy()
        
        # Process audio with USM feature extractor
        inputs = self.feature_extractor(
            audio_np,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        # Move inputs to device
        audio_mel = inputs["input_features"].to(self.device)
        audio_mel_mask = (inputs["input_features_mask"] == 0).to(self.device)
        
        # Extract features with USM encoder
        with torch.no_grad() if self.freeze else torch.enable_grad():
            audio_encodings, output_mask = self.audio_encoder(
                audio_mel=audio_mel,
                audio_mel_mask=audio_mel_mask
            )
        
        return audio_encodings
    
    
    def get_feature_dim(self) -> int:
        """Get the feature dimension of USM features."""
        # Get actual feature dimension from USM model
        try:
            dummy_audio = torch.randn(1, 16000).to(self.device)
            with torch.no_grad():
                features = self.extract_features(dummy_audio)
                return features.shape[-1]
        except:
            # Default feature dimension for Atotti/Google-USM
            return 1536
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the USM model."""
        return {
            "model_name": self.model_name,
            "source_model_id": self.source_model_id,
            "frozen": self.freeze,
            "device": self.device,
            "feature_dim": self.get_feature_dim(),
            "architecture": "Gemma3nAudioEncoder"
        }


class USMConfig:
    """Configuration for USM model."""
    
    # Available USM models
    AVAILABLE_MODELS = {
        "Atotti/google-usm": {
            "feature_dim": 1536,
            "max_audio_length": 30.0,  # seconds
            "sample_rate": 16000
        },
        "Atotti/google-usm-bf16": {
            "feature_dim": 1536,
            "max_audio_length": 30.0,
            "sample_rate": 16000
        }
    }
    
    @classmethod
    def get_model_config(cls, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific USM model."""
        return cls.AVAILABLE_MODELS.get(model_name, {
            "feature_dim": 1536,
            "max_audio_length": 30.0,
            "sample_rate": 16000
        })
    
    @classmethod
    def list_available_models(cls) -> list:
        """List available USM models."""
        return list(cls.AVAILABLE_MODELS.keys())


def create_usm_extractor(
    model_name: str = "Atotti/google-usm",
    freeze: bool = True,
    device: Optional[str] = None
) -> USMFeatureExtractor:
    """
    Factory function to create USM feature extractor.
    
    Args:
        model_name: Name of USM model to use
        freeze: Whether to freeze USM parameters
        device: Device to run on
        
    Returns:
        USMFeatureExtractor instance
    """
    return USMFeatureExtractor(
        model_name=model_name,
        freeze=freeze,
        device=device
    )


# Example usage
if __name__ == "__main__":
    # Create USM extractor
    usm_extractor = create_usm_extractor(
        model_name="Atotti/google-usm",
        freeze=True
    )
    
    # Test with dummy audio
    dummy_audio = torch.randn(2, 16000)  # 2 seconds of audio
    features = usm_extractor.extract_features(dummy_audio)
    
    print(f"USM features shape: {features.shape}")
    print(f"Model info: {usm_extractor.get_model_info()}")
