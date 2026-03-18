"""Unit tests for data preprocessing."""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.preprocessor import DataProcessor


class TestDataProcessor:
    """Tests for DataProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create a DataProcessor instance."""
        return DataProcessor()
    
    def test_load_sample_data(self, processor):
        """Test loading sample Iris data."""
        df = processor.load_sample_data()
        
        assert df is not None
        assert len(df) == 150
        assert 'species' in df.columns
        assert 'species_name' in df.columns
    
    def test_clean_data(self, processor):
        """Test data cleaning."""
        processor.load_sample_data()
        cleaned = processor.clean_data()
        
        assert cleaned is not None
        assert not cleaned.isnull().any().any()
    
    def test_get_features_and_target(self, processor):
        """Test feature/target separation."""
        processor.load_sample_data()
        processor.clean_data()
        X, y = processor.get_features_and_target(target_col='species')
        
        assert X.shape[0] == len(y)
        assert len(processor.feature_columns) == 4
    
    def test_split_data(self, processor):
        """Test train/test split."""
        processor.load_sample_data()
        processor.clean_data()
        X, y = processor.get_features_and_target()
        X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2)
        
        assert len(X_train) == 120
        assert len(X_test) == 30
    
    def test_fit_transform(self, processor):
        """Test feature scaling."""
        processor.load_sample_data()
        processor.clean_data()
        X, _ = processor.get_features_and_target()
        
        X_scaled = processor.fit_transform(X)
        
        assert X_scaled.shape == X.shape
        # Check standardization (mean ~0, std ~1)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-10)
    
    def test_transform_without_fit(self, processor):
        """Test transform raises error without fit."""
        processor.load_sample_data()
        processor.clean_data()
        X, _ = processor.get_features_and_target()
        
        with pytest.raises(Exception):
            processor.transform(X)
