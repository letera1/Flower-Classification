"""FastAPI backend for flower classification."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import numpy as np
import joblib
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Flower Classification API",
    description="API for classifying flower species based on morphological features",
    version="1.0.0"
)

model = None
preprocessor = None
class_names = None
feature_columns = None


class FlowerFeatures(BaseModel):
    """Input features for flower classification."""
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0)
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0)
    petal_length: float = Field(..., description="Petal length in cm", ge=0)
    petal_width: float = Field(..., description="Petal width in cm", ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }


class PredictionResponse(BaseModel):
    """Response from prediction endpoint."""
    species: str
    species_id: int
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    timestamp: str


def load_model(model_path: str = "models/best_model.joblib",
               preprocessor_path: str = "models/preprocessor.joblib",
               metadata_path: str = "models/metadata.json"):
    """Load the trained model and preprocessor."""
    global model, preprocessor, class_names, feature_columns
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        preprocessor = joblib.load(preprocessor_path)
        logger.info(f"Preprocessor loaded from {preprocessor_path}")
        
        import json
        if Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                class_names = metadata.get('class_names')
                feature_columns = metadata.get('feature_columns')
                logger.info(f"Loaded metadata: {len(class_names)} classes")
        else:
            class_names = ['setosa', 'versicolor', 'virginica']
            feature_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            
    except FileNotFoundError as e:
        logger.warning(f"Model files not found: {e}")
        logger.info("API will run in demo mode")
        return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False
    
    return True


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    load_model()


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Flower Classification API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        timestamp=datetime.utcnow().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: FlowerFeatures):
    """Predict flower species from input features."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_array = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        if preprocessor is not None:
            input_scaled = preprocessor.transform(input_array)
        else:
            input_scaled = input_array
        
        prediction = model.predict(input_scaled)[0]
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            if class_names:
                probabilities = {name: float(p) for name, p in zip(class_names, proba)}
        
        confidence = float(np.max(model.predict_proba(input_scaled))) if hasattr(model, 'predict_proba') else 1.0
        
        return PredictionResponse(
            species=class_names[prediction] if class_names else f"class_{prediction}",
            species_id=int(prediction),
            confidence=confidence,
            probabilities=probabilities,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(features_list: List[FlowerFeatures]):
    """Batch prediction for multiple flower samples."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    for features in features_list:
        input_array = np.array([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]])
        
        if preprocessor is not None:
            input_scaled = preprocessor.transform(input_array)
        else:
            input_scaled = input_array
        
        prediction = model.predict(input_scaled)[0]
        
        probabilities = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_scaled)[0]
            if class_names:
                probabilities = {name: float(p) for name, p in zip(class_names, proba)}
        
        confidence = float(np.max(model.predict_proba(input_scaled))) if hasattr(model, 'predict_proba') else 1.0
        
        results.append(PredictionResponse(
            species=class_names[prediction] if class_names else f"class_{prediction}",
            species_id=int(prediction),
            confidence=confidence,
            probabilities=probabilities,
            timestamp=datetime.utcnow().isoformat()
        ))
    
    return results


@app.get("/classes")
async def get_classes():
    """Get list of flower classes the model can predict."""
    if class_names:
        return {"classes": class_names, "count": len(class_names)}
    return {"classes": [], "count": 0, "message": "Model not loaded"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
