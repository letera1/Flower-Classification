"""Unit tests for model training."""
import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.trainer import ModelTrainer


class TestModelTrainer:
    """Tests for ModelTrainer class."""
    
    @pytest.fixture
    def trainer(self):
        """Create a ModelTrainer instance."""
        return ModelTrainer(random_state=42)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)
        return X, y
    
    def test_initialize_models(self, trainer):
        """Test model initialization."""
        models = trainer.initialize_models()
        
        assert len(models) == 4
        assert 'logistic_regression' in models
        assert 'decision_tree' in models
        assert 'random_forest' in models
        assert 'xgboost' in models
    
    def test_train_model(self, trainer, sample_data):
        """Test training a single model."""
        trainer.initialize_models()
        X, y = sample_data
        
        model = trainer.train_model('logistic_regression', X, y)
        
        assert model is not None
        assert hasattr(model, 'predict')
    
    def test_train_all(self, trainer, sample_data):
        """Test training all models."""
        trainer.initialize_models()
        X, y = sample_data
        
        models = trainer.train_all(X, y)
        
        assert len(models) == 4
        for name, model in models.items():
            assert model is not None
    
    def test_predict(self, trainer, sample_data):
        """Test model prediction."""
        trainer.initialize_models()
        X, y = sample_data
        trainer.train_all(X, y)
        
        predictions = trainer.models['logistic_regression'].predict(X[:5])
        
        assert len(predictions) == 5
        assert all(p in [0, 1, 2] for p in predictions)
    
    def test_get_feature_importance(self, trainer, sample_data):
        """Test feature importance extraction."""
        trainer.initialize_models()
        X, y = sample_data
        trainer.train_all(X, y)
        
        importance = trainer.get_feature_importance('random_forest')
        
        assert importance is not None
        assert len(importance) == 4
    
    def test_invalid_model_name(self, trainer):
        """Test error handling for invalid model."""
        with pytest.raises(ValueError):
            trainer.train_model('invalid_model', np.array([]), np.array([]))
