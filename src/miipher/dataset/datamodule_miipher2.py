from typing import Any, Dict, Optional, List
import torch
from torch.utils.data import DataLoader, Dataset
from lightning.pytorch import LightningDataModule
from omegaconf import DictConfig
import webdataset as wds
import logging
import random


class Miipher2Dataset(Dataset):
    """
    Dataset for Miipher-2 training.
    Conditioning-free: only requires noisy and clean audio pairs.
    """
    
    def __init__(
        self,
        dataset_path: str,
        sample_rate: int = 22050,
        max_audio_length: float = 10.0,
        min_audio_length: float = 1.0,
        cache_size: int = 1000
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.cache_size = cache_size
        
        # Convert to samples
        self.max_samples = int(max_audio_length * sample_rate)
        self.min_samples = int(min_audio_length * sample_rate)
        
        # Initialize dataset
        self.dataset = self._create_dataset()
        
        logging.info(f"Miipher2Dataset initialized with {len(self.dataset)} samples")
    
    def _create_dataset(self):
        """Create webdataset from tar files."""
        dataset = (
            wds.WebDataset(self.dataset_path)
            .decode("torch")
            .map(self._process_sample)
            .filter(self._filter_sample)
            .to_tuple("noisy_audio", "clean_audio", "audio_length")
        )
        return dataset
    
    def _process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sample from the dataset."""
        # Extract audio data
        noisy_audio = sample["noisy_audio.pth"]
        clean_audio = sample["clean_audio.pth"]
        
        # Ensure audio is float32 and normalized
        if noisy_audio.dtype != torch.float32:
            noisy_audio = noisy_audio.float()
        if clean_audio.dtype != torch.float32:
            clean_audio = clean_audio.float()
        
        # Normalize audio to [-1, 1] range
        noisy_audio = torch.clamp(noisy_audio, -1.0, 1.0)
        clean_audio = torch.clamp(clean_audio, -1.0, 1.0)
        
        # Get audio length
        audio_length = len(clean_audio)
        
        return {
            "noisy_audio": noisy_audio,
            "clean_audio": clean_audio,
            "audio_length": audio_length
        }
    
    def _filter_sample(self, sample: Dict[str, Any]) -> bool:
        """Filter samples based on length and quality criteria."""
        audio_length = sample["audio_length"]
        
        # Filter by length
        if audio_length < self.min_samples or audio_length > self.max_samples:
            return False
        
        # Additional quality filters can be added here
        # e.g., check for silence, clipping, etc.
        
        return True
    
    def __len__(self) -> int:
        """Return dataset length (approximate for webdataset)."""
        # For webdataset, we can't get exact length without iterating
        # Return a large number to indicate it's a streaming dataset
        return 1000000  # Placeholder
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset."""
        # For webdataset, we iterate through the dataset
        # This is a simplified implementation
        for sample in self.dataset:
            return sample
        raise IndexError("Dataset exhausted")


class Miipher2DataModule(LightningDataModule):
    """
    Lightning DataModule for Miipher-2.
    Handles conditioning-free data loading and preprocessing.
    """
    
    def __init__(
        self,
        train_dataset_path: str,
        val_dataset_path: str,
        train_batch_size: int = 16,
        val_batch_size: int = 16,
        num_workers: int = 4,
        sample_rate: int = 22050,
        max_audio_length: float = 10.0,
        min_audio_length: float = 1.0,
        pin_memory: bool = True,
        persistent_workers: bool = True
    ):
        super().__init__()
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        
        # Datasets
        self.train_dataset = None
        self.val_dataset = None
        
        logging.info("Miipher2DataModule initialized")
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets for training/validation."""
        if stage == "fit" or stage is None:
            self.train_dataset = Miipher2Dataset(
                dataset_path=self.train_dataset_path,
                sample_rate=self.sample_rate,
                max_audio_length=self.max_audio_length,
                min_audio_length=self.min_audio_length
            )
            
            self.val_dataset = Miipher2Dataset(
                dataset_path=self.val_dataset_path,
                sample_rate=self.sample_rate,
                max_audio_length=self.max_audio_length,
                min_audio_length=self.min_audio_length
            )
        
        if stage == "test" or stage is None:
            # For testing, we can use the validation dataset
            # or create a separate test dataset
            self.test_dataset = Miipher2Dataset(
                dataset_path=self.val_dataset_path,  # Using val as test for now
                sample_rate=self.sample_rate,
                max_audio_length=self.max_audio_length,
                min_audio_length=self.min_audio_length
            )
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn,
            drop_last=True
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn,
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            collate_fn=self._collate_fn,
            drop_last=False
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collate function for batching samples.
        Handles variable length audio sequences.
        """
        noisy_audios = []
        clean_audios = []
        audio_lengths = []
        
        for sample in batch:
            noisy_audios.append(sample["noisy_audio"])
            clean_audios.append(sample["clean_audio"])
            audio_lengths.append(sample["audio_length"])
        
        # Pad sequences to the same length
        max_length = max(audio_lengths)
        
        # Pad noisy audio
        padded_noisy = []
        for audio in noisy_audios:
            if len(audio) < max_length:
                padding = torch.zeros(max_length - len(audio))
                padded_audio = torch.cat([audio, padding])
            else:
                padded_audio = audio[:max_length]
            padded_noisy.append(padded_audio)
        
        # Pad clean audio
        padded_clean = []
        for audio in clean_audios:
            if len(audio) < max_length:
                padding = torch.zeros(max_length - len(audio))
                padded_audio = torch.cat([audio, padding])
            else:
                padded_audio = audio[:max_length]
            padded_clean.append(padded_audio)
        
        return {
            "noisy_audio": torch.stack(padded_noisy),
            "clean_audio": torch.stack(padded_clean),
            "audio_lengths": torch.tensor(audio_lengths, dtype=torch.long)
        }


class MultilingualMiipher2Dataset(Miipher2Dataset):
    """
    Extended dataset for multilingual Miipher-2 training.
    Supports multiple languages without explicit conditioning.
    """
    
    def __init__(
        self,
        dataset_paths: List[str],
        languages: List[str],
        language_weights: Optional[List[float]] = None,
        **kwargs
    ):
        self.dataset_paths = dataset_paths
        self.languages = languages
        self.language_weights = language_weights or [1.0] * len(languages)
        
        # Normalize weights
        total_weight = sum(self.language_weights)
        self.language_weights = [w / total_weight for w in self.language_weights]
        
        # Create datasets for each language
        self.language_datasets = []
        for dataset_path in dataset_paths:
            dataset = Miipher2Dataset(dataset_path, **kwargs)
            self.language_datasets.append(dataset)
        
        super().__init__(dataset_paths[0], **kwargs)
        
        logging.info(f"MultilingualMiipher2Dataset initialized with {len(languages)} languages")
        logging.info(f"Languages: {languages}")
        logging.info(f"Language weights: {self.language_weights}")
    
    def _select_language_dataset(self) -> Miipher2Dataset:
        """Select a dataset based on language weights."""
        # Sample language based on weights
        selected_idx = random.choices(
            range(len(self.language_datasets)),
            weights=self.language_weights,
            k=1
        )[0]
        
        return self.language_datasets[selected_idx]
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from a randomly selected language dataset."""
        selected_dataset = self._select_language_dataset()
        return next(iter(selected_dataset))


class Miipher2Preprocessor:
    """
    Preprocessor for Miipher-2 data preparation.
    Handles audio preprocessing without conditioning requirements.
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        max_audio_length: float = 10.0,
        min_audio_length: float = 1.0,
        normalize: bool = True
    ):
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.normalize = normalize
        
        self.max_samples = int(max_audio_length * sample_rate)
        self.min_samples = int(min_audio_length * sample_rate)
    
    def preprocess_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Preprocess audio for Miipher-2 training.
        
        Args:
            audio: Input audio tensor
        Returns:
            processed_audio: Preprocessed audio tensor
        """
        # Ensure float32
        if audio.dtype != torch.float32:
            audio = audio.float()
        
        # Normalize to [-1, 1] range
        if self.normalize:
            audio = torch.clamp(audio, -1.0, 1.0)
        
        # Truncate or pad to target length
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        elif len(audio) < self.min_samples:
            padding = torch.zeros(self.min_samples - len(audio))
            audio = torch.cat([audio, padding])
        
        return audio
    
    def create_noisy_clean_pair(
        self, 
        clean_audio: torch.Tensor,
        degradation_config: Dict[str, Any]
    ) -> torch.Tensor:
        """
        Create noisy-clean audio pair for training.
        
        Args:
            clean_audio: Clean reference audio
            degradation_config: Configuration for degradation
        Returns:
            noisy_audio: Degraded version of clean audio
        """
        # This is a placeholder implementation
        # In practice, this would apply various degradations:
        # - Compression artifacts
        # - Background noise
        # - Reverberation
        # - etc.
        
        noisy_audio = clean_audio.clone()
        
        # Add simple noise for demonstration
        noise_level = degradation_config.get('noise_level', 0.1)
        noise = torch.randn_like(clean_audio) * noise_level
        noisy_audio = noisy_audio + noise
        
        # Clamp to valid range
        noisy_audio = torch.clamp(noisy_audio, -1.0, 1.0)
        
        return noisy_audio