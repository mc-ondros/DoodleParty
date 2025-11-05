"""Unit tests for data loading and preprocessing."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil

from src.data.loaders import QuickDrawDataset


class TestQuickDrawDataset:
    """Test QuickDrawDataset class."""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def dataset(self, temp_data_dir):
        """Create QuickDrawDataset instance with temp directory."""
        return QuickDrawDataset(data_dir=str(temp_data_dir))
    
    def test_init_creates_directory(self, temp_data_dir):
        """Test that initialization creates data directory."""
        new_dir = temp_data_dir / "new_data"
        dataset = QuickDrawDataset(data_dir=str(new_dir))
        assert new_dir.exists()
        assert dataset.data_dir == new_dir
    
    def test_download_class_skips_existing(self, dataset, temp_data_dir, capsys):
        """Test that download skips already downloaded files."""
        # Create dummy file
        test_file = temp_data_dir / "airplane.npy"
        np.save(test_file, np.random.rand(100, 28, 28))
        
        dataset.download_class("airplane")
        captured = capsys.readouterr()
        assert "already downloaded" in captured.out
    
    @patch('src.data.loaders.requests.get')
    def test_download_class_success(self, mock_get, dataset, temp_data_dir):
        """Test successful class download."""
        # Mock response
        mock_response = Mock()
        mock_response.headers = {'content-length': '1024'}
        mock_response.iter_content = lambda chunk_size: [b'test_data']
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        dataset.download_class("test_class")
        
        # Verify file was created
        assert (temp_data_dir / "test_class.npy").exists()
    
    @patch('src.data.loaders.requests.get')
    def test_download_class_handles_error(self, mock_get, dataset, temp_data_dir, capsys):
        """Test error handling during download."""
        mock_get.side_effect = Exception("Network error")
        
        dataset.download_class("test_class")
        captured = capsys.readouterr()
        assert "Error downloading" in captured.out
    
    def test_load_class_data_file_not_found(self, dataset):
        """Test error when loading non-existent class."""
        with pytest.raises(FileNotFoundError):
            dataset.load_class_data("nonexistent_class")
    
    def test_load_class_data_success(self, dataset, temp_data_dir):
        """Test successful data loading."""
        # Create test data
        test_data = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
        np.save(temp_data_dir / "airplane.npy", test_data)
        
        loaded_data = dataset.load_class_data("airplane")
        
        assert loaded_data.shape == (100, 28, 28)
        assert loaded_data.dtype == np.float32
        assert loaded_data.min() >= 0.0
        assert loaded_data.max() <= 1.0
    
    def test_load_class_data_max_samples(self, dataset, temp_data_dir):
        """Test loading with max_samples limit."""
        test_data = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
        np.save(temp_data_dir / "airplane.npy", test_data)
        
        loaded_data = dataset.load_class_data("airplane", max_samples=50)
        
        assert loaded_data.shape == (50, 28, 28)
    
    def test_load_class_data_no_normalize(self, dataset, temp_data_dir):
        """Test loading without normalization."""
        test_data = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
        np.save(temp_data_dir / "airplane.npy", test_data)
        
        loaded_data = dataset.load_class_data("airplane", normalize=False)
        
        assert loaded_data.dtype == np.uint8
        assert loaded_data.max() > 1.0
    
    def test_prepare_dataset(self, dataset, temp_data_dir):
        """Test binary dataset preparation."""
        # Create test data for positive classes
        for class_name in ["airplane", "apple"]:
            test_data = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
            np.save(temp_data_dir / f"{class_name}.npy", test_data)
        
        output_dir = temp_data_dir / "processed"
        (X_train, y_train), (X_test, y_test), class_mapping = dataset.prepare_dataset(
            classes=["airplane", "apple"],
            output_dir=str(output_dir),
            max_samples_per_class=100,
            test_split=0.2
        )
        
        # Verify shapes
        total_samples = 400  # 100 airplane + 100 apple + 200 negative
        train_samples = int(total_samples * 0.8)
        test_samples = total_samples - train_samples
        
        assert X_train.shape == (train_samples, 28, 28, 1)
        assert y_train.shape == (train_samples,)
        assert X_test.shape == (test_samples, 28, 28, 1)
        assert y_test.shape == (test_samples,)
        
        # Verify labels
        assert set(np.unique(y_train)) == {0, 1}
        assert set(np.unique(y_test)) == {0, 1}
        
        # Verify class balance (approximately 50/50)
        positive_ratio = (y_train == 1).sum() / len(y_train)
        assert 0.4 < positive_ratio < 0.6
        
        # Verify files saved
        assert (output_dir / "X_train.npy").exists()
        assert (output_dir / "y_train.npy").exists()
        assert (output_dir / "X_test.npy").exists()
        assert (output_dir / "y_test.npy").exists()
        assert (output_dir / "class_mapping.pkl").exists()
        
        # Verify class mapping
        assert class_mapping['negative'] == 0
        assert class_mapping['positive'] == 1
        assert class_mapping['positive_classes'] == ["airplane", "apple"]
    
    def test_prepare_dataset_normalization(self, dataset, temp_data_dir):
        """Test that prepared dataset is properly normalized."""
        # Create test data
        test_data = np.random.randint(0, 256, (50, 28, 28), dtype=np.uint8)
        np.save(temp_data_dir / "airplane.npy", test_data)
        
        output_dir = temp_data_dir / "processed"
        (X_train, y_train), (X_test, y_test), _ = dataset.prepare_dataset(
            classes=["airplane"],
            output_dir=str(output_dir),
            max_samples_per_class=50,
            test_split=0.2
        )
        
        # Verify normalization
        assert X_train.dtype == np.float32
        assert X_train.min() >= 0.0
        assert X_train.max() <= 1.0
        assert X_test.min() >= 0.0
        assert X_test.max() <= 1.0
