"""Prediction service for flower classification."""
import numpy as np
import joblib
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging
from functools import lru_cache

from app.core.config import get_settings, ConfigLoader

logger = logging.getLogger(__name__)


class PredictorService:
    """Service for loading models and making predictions."""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.class_names: List[str] = []
        self.feature_columns: List[str] = []
        self.model_info: Dict[str, Any] = {}
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load model and preprocessor from disk."""
        settings = get_settings()
        config = ConfigLoader().load()
        
        # Determine model path
        model_path = settings.MODELS_DIR / "best_model.joblib"
        preprocessor_path = settings.MODELS_DIR / "preprocessor.joblib"
        metadata_path = settings.MODELS_DIR / "metadata.json"
        
        try:
            # Load model
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model not found at {model_path}")
                return False
            
            # Load preprocessor
            if preprocessor_path.exists():
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info(f"Preprocessor loaded from {preprocessor_path}")
            else:
                logger.warning(f"Preprocessor not found at {preprocessor_path}")
                return False
            
            # Load metadata
            import json
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.class_names = metadata.get('class_names', [])
                    self.feature_columns = metadata.get('feature_columns', [])
                    self.model_info = metadata
                logger.info(f"Loaded metadata: {len(self.class_names)} classes")
            else:
                # Default for Iris dataset
                self.class_names = ['setosa', 'versicolor', 'virginica']
                self.feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
                logger.info("Using default class names")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded and ready."""
        return self.model is not None and self.preprocessor is not None
    
    def predict_single(
        self,
        sepal_length: float,
        sepal_width: float,
        petal_length: float,
        petal_width: float
    ) -> Dict[str, Any]:
        """Make a single prediction."""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        # Prepare input
        input_array = np.array([[
            sepal_length,
            sepal_width,
            petal_length,
            petal_width
        ]])
        
        # Scale features
        input_scaled = self.preprocessor.transform(input_array)
        
        # Predict
        prediction = self.model.predict(input_scaled)[0]
        
        # Get probabilities
        probabilities = None
        confidence = 1.0
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(input_scaled)[0]
            probabilities = {
                name: float(p) 
                for name, p in zip(self.class_names, proba)
            }
            confidence = float(np.max(proba))
        
        return {
            "species": self.class_names[prediction] if self.class_names else f"class_{prediction}",
            "species_id": int(prediction),
            "confidence": confidence,
            "probabilities": probabilities
        }
    
    def predict_batch(
        self,
        samples: List[Dict[str, float]]
    ) -> List[Dict[str, Any]]:
        """Make batch predictions."""
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        # Prepare input
        input_array = np.array([
            [
                s['sepal_length'],
                s['sepal_width'],
                s['petal_length'],
                s['petal_width']
            ]
            for s in samples
        ])
        
        # Scale features
        input_scaled = self.preprocessor.transform(input_array)
        
        # Predict
        predictions = self.model.predict(input_scaled)
        
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "species": self.class_names[pred] if self.class_names else f"class_{pred}",
                "species_id": int(pred),
                "confidence": 1.0
            }
            
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(input_scaled)[i]
                result["probabilities"] = {
                    name: float(p) 
                    for name, p in zip(self.class_names, proba)
                }
                result["confidence"] = float(np.max(proba))
            
            results.append(result)
        
        return results
    
    def get_classes(self) -> List[str]:
        """Get list of class names."""
        return self.class_names
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get model information."""
        if not self.is_model_loaded():
            return None
        
        return {
            "model_name": self.model_info.get("model_name", "unknown"),
            "classes": self.class_names,
            "features": self.feature_columns,
            "accuracy": self.model_info.get("metrics", {}).get("accuracy"),
            "training_samples": self.model_info.get("training_samples"),
            "test_samples": self.model_info.get("test_samples")
        }


@lru_cache()
def get_predictor() -> PredictorService:
    """Get cached predictor instance."""
    return PredictorService()
