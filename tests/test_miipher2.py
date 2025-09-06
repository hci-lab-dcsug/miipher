import unittest
import torch
import tempfile
import os
from miipher.model.miipher2 import (
    ParallelAdapter, 
    Miipher2, 
    USMFeatureExtractor, 
    Miipher2Complete
)
from miipher.dataset.datamodule_miipher2 import (
    Miipher2Dataset,
    Miipher2DataModule,
    MultilingualMiipher2Dataset,
    Miipher2Preprocessor
)


class TestParallelAdapter(unittest.TestCase):
    """Test cases for ParallelAdapter module."""
    
    def setUp(self):
        self.usm_dim = 1024
        self.hidden_dim = 512
        self.batch_size = 2
        self.seq_len = 100
        
        self.adapter = ParallelAdapter(
            usm_dim=self.usm_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.1
        )
    
    def test_parallel_adapter_forward(self):
        """Test forward pass of ParallelAdapter."""
        # Create input tensor
        noisy_features = torch.randn(self.batch_size, self.seq_len, self.usm_dim)
        
        # Forward pass
        clean_features = self.adapter(noisy_features)
        
        # Check output shape
        self.assertEqual(clean_features.shape, noisy_features.shape)
        self.assertEqual(clean_features.shape, (self.batch_size, self.seq_len, self.usm_dim))
    
    def test_parallel_adapter_residual_connection(self):
        """Test that residual connection is properly implemented."""
        noisy_features = torch.randn(self.batch_size, self.seq_len, self.usm_dim)
        
        # Forward pass
        clean_features = self.adapter(noisy_features)
        
        # Check that output is different from input (adapter is working)
        self.assertFalse(torch.allclose(clean_features, noisy_features))
        
        # Check that residual connection is present
        # (output should be input + adapter_output)
        adapter_output = clean_features - noisy_features
        self.assertFalse(torch.allclose(adapter_output, torch.zeros_like(adapter_output)))
    
    def test_parallel_adapter_gradient_flow(self):
        """Test that gradients flow properly through the adapter."""
        noisy_features = torch.randn(self.batch_size, self.seq_len, self.usm_dim, requires_grad=True)
        
        # Forward pass
        clean_features = self.adapter(noisy_features)
        
        # Compute loss and backward pass
        loss = clean_features.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(noisy_features.grad)
        self.assertFalse(torch.allclose(noisy_features.grad, torch.zeros_like(noisy_features.grad)))


class TestMiipher2(unittest.TestCase):
    """Test cases for Miipher2 model."""
    
    def setUp(self):
        self.usm_dim = 1024
        self.n_adapters = 6
        self.adapter_hidden_dim = 512
        self.batch_size = 2
        self.seq_len = 100
        
        self.miipher2 = Miipher2(
            usm_dim=self.usm_dim,
            n_adapters=self.n_adapters,
            adapter_hidden_dim=self.adapter_hidden_dim,
            dropout=0.1
        )
    
    def test_miipher2_forward(self):
        """Test forward pass of Miipher2."""
        # Create input tensor
        noisy_usm_features = torch.randn(self.batch_size, self.seq_len, self.usm_dim)
        
        # Forward pass
        clean_features = self.miipher2(noisy_usm_features)
        
        # Check output shape
        self.assertEqual(clean_features.shape, noisy_usm_features.shape)
        self.assertEqual(clean_features.shape, (self.batch_size, self.seq_len, self.usm_dim))
    
    def test_miipher2_with_lengths(self):
        """Test Miipher2 with sequence lengths."""
        noisy_usm_features = torch.randn(self.batch_size, self.seq_len, self.usm_dim)
        feature_lengths = torch.tensor([self.seq_len, self.seq_len - 10])
        
        # Forward pass
        clean_features = self.miipher2(noisy_usm_features, feature_lengths)
        
        # Check output shape
        self.assertEqual(clean_features.shape, noisy_usm_features.shape)
    
    def test_miipher2_parameter_count(self):
        """Test parameter counting functionality."""
        param_info = self.miipher2.get_num_parameters()
        
        # Check that parameter info is returned
        self.assertIn('total_parameters', param_info)
        self.assertIn('trainable_parameters', param_info)
        self.assertIn('adapter_parameters', param_info)
        self.assertIn('efficiency_ratio', param_info)
        
        # Check that parameters are reasonable
        self.assertGreater(param_info['total_parameters'], 0)
        self.assertGreater(param_info['trainable_parameters'], 0)
        self.assertGreater(param_info['adapter_parameters'], 0)
        self.assertGreaterEqual(param_info['efficiency_ratio'], 0.0)
        self.assertLessEqual(param_info['efficiency_ratio'], 1.0)


class TestUSMFeatureExtractor(unittest.TestCase):
    """Test cases for USMFeatureExtractor."""
    
    def setUp(self):
        self.usm_model_name = "google/usm-large"  # Placeholder
        self.batch_size = 2
        self.audio_length = 16000  # 1 second at 16kHz
        
        self.usm_extractor = USMFeatureExtractor(
            usm_model_name=self.usm_model_name,
            freeze=True
        )
    
    def test_usm_feature_extractor_forward(self):
        """Test forward pass of USMFeatureExtractor."""
        # Create input audio
        audio = torch.randn(self.batch_size, self.audio_length)
        
        # Forward pass
        usm_features = self.usm_extractor(audio)
        
        # Check output shape
        self.assertEqual(len(usm_features.shape), 3)  # (batch_size, seq_len, usm_dim)
        self.assertEqual(usm_features.shape[0], self.batch_size)
    
    def test_usm_feature_extractor_frozen(self):
        """Test that USM parameters are frozen."""
        # Check that USM parameters don't require gradients
        for param in self.usm_extractor.parameters():
            self.assertFalse(param.requires_grad)


class TestMiipher2Complete(unittest.TestCase):
    """Test cases for complete Miipher2 system."""
    
    def setUp(self):
        self.usm_model_name = "google/usm-large"  # Placeholder
        self.usm_dim = 1024
        self.n_adapters = 6
        self.adapter_hidden_dim = 512
        self.batch_size = 2
        self.audio_length = 16000
        
        self.miipher2_complete = Miipher2Complete(
            usm_model_name=self.usm_model_name,
            usm_dim=self.usm_dim,
            n_adapters=self.n_adapters,
            adapter_hidden_dim=self.adapter_hidden_dim,
            freeze_usm=True
        )
    
    def test_miipher2_complete_forward(self):
        """Test complete forward pass."""
        # Create input audio
        noisy_audio = torch.randn(self.batch_size, self.audio_length)
        
        # Forward pass
        clean_features = self.miipher2_complete(noisy_audio)
        
        # Check output shape
        self.assertEqual(len(clean_features.shape), 3)  # (batch_size, seq_len, usm_dim)
        self.assertEqual(clean_features.shape[0], self.batch_size)
    
    def test_miipher2_complete_model_info(self):
        """Test model info functionality."""
        model_info = self.miipher2_complete.get_model_info()
        
        # Check that model info is returned
        self.assertIn('model_name', model_info)
        self.assertIn('usm_model', model_info)
        self.assertIn('usm_frozen', model_info)
        self.assertIn('total_params', model_info)
        self.assertIn('trainable_params', model_info)
        
        # Check that it's Miipher-2
        self.assertEqual(model_info['model_name'], 'Miipher-2')


class TestMiipher2Preprocessor(unittest.TestCase):
    """Test cases for Miipher2Preprocessor."""
    
    def setUp(self):
        self.sample_rate = 22050
        self.max_audio_length = 10.0
        self.min_audio_length = 1.0
        
        self.preprocessor = Miipher2Preprocessor(
            sample_rate=self.sample_rate,
            max_audio_length=self.max_audio_length,
            min_audio_length=self.min_audio_length,
            normalize=True
        )
    
    def test_preprocess_audio(self):
        """Test audio preprocessing."""
        # Create test audio
        audio_length = int(5.0 * self.sample_rate)  # 5 seconds
        audio = torch.randn(audio_length) * 0.5  # Small amplitude
        
        # Preprocess
        processed_audio = self.preprocessor.preprocess_audio(audio)
        
        # Check output properties
        self.assertEqual(processed_audio.dtype, torch.float32)
        self.assertTrue(torch.all(processed_audio >= -1.0))
        self.assertTrue(torch.all(processed_audio <= 1.0))
    
    def test_preprocess_audio_length_handling(self):
        """Test audio length handling (truncation and padding)."""
        # Test with audio that's too long
        long_audio = torch.randn(int(15.0 * self.sample_rate))  # 15 seconds
        processed_long = self.preprocessor.preprocess_audio(long_audio)
        expected_max_length = int(self.max_audio_length * self.sample_rate)
        self.assertEqual(len(processed_long), expected_max_length)
        
        # Test with audio that's too short
        short_audio = torch.randn(int(0.5 * self.sample_rate))  # 0.5 seconds
        processed_short = self.preprocessor.preprocess_audio(short_audio)
        expected_min_length = int(self.min_audio_length * self.sample_rate)
        self.assertEqual(len(processed_short), expected_min_length)
    
    def test_create_noisy_clean_pair(self):
        """Test noisy-clean pair creation."""
        # Create clean audio
        clean_audio = torch.randn(int(5.0 * self.sample_rate))
        
        # Create degradation config
        degradation_config = {'noise_level': 0.1}
        
        # Create noisy-clean pair
        noisy_audio = self.preprocessor.create_noisy_clean_pair(
            clean_audio, degradation_config
        )
        
        # Check output properties
        self.assertEqual(noisy_audio.shape, clean_audio.shape)
        self.assertTrue(torch.all(noisy_audio >= -1.0))
        self.assertTrue(torch.all(noisy_audio <= 1.0))
        
        # Check that noisy audio is different from clean
        self.assertFalse(torch.allclose(noisy_audio, clean_audio))


class TestMiipher2Dataset(unittest.TestCase):
    """Test cases for Miipher2Dataset."""
    
    def setUp(self):
        # Create temporary dataset for testing
        self.temp_dir = tempfile.mkdtemp()
        self.dataset_path = os.path.join(self.temp_dir, "test_dataset.tar")
        
        # Create a simple test dataset
        self._create_test_dataset()
        
        self.dataset = Miipher2Dataset(
            dataset_path=self.dataset_path,
            sample_rate=22050,
            max_audio_length=5.0,
            min_audio_length=1.0
        )
    
    def _create_test_dataset(self):
        """Create a simple test dataset."""
        # This is a placeholder - in practice, you'd create actual tar files
        # with audio data for testing
        pass
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        self.assertIsNotNone(self.dataset)
        self.assertEqual(self.dataset.sample_rate, 22050)
        self.assertEqual(self.dataset.max_audio_length, 5.0)
        self.assertEqual(self.dataset.min_audio_length, 1.0)


class TestMiipher2DataModule(unittest.TestCase):
    """Test cases for Miipher2DataModule."""
    
    def setUp(self):
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.train_path = os.path.join(self.temp_dir, "train.tar")
        self.val_path = os.path.join(self.temp_dir, "val.tar")
        
        # Create test datasets
        self._create_test_datasets()
        
        self.data_module = Miipher2DataModule(
            train_dataset_path=self.train_path,
            val_dataset_path=self.val_path,
            train_batch_size=4,
            val_batch_size=4,
            num_workers=0,  # Use 0 for testing
            sample_rate=22050,
            max_audio_length=5.0,
            min_audio_length=1.0
        )
    
    def _create_test_datasets(self):
        """Create test datasets."""
        # This is a placeholder - in practice, you'd create actual tar files
        pass
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_data_module_initialization(self):
        """Test data module initialization."""
        self.assertIsNotNone(self.data_module)
        self.assertEqual(self.data_module.train_batch_size, 4)
        self.assertEqual(self.data_module.val_batch_size, 4)
        self.assertEqual(self.data_module.sample_rate, 22050)


if __name__ == '__main__':
    # Run all tests
    unittest.main(verbosity=2)
