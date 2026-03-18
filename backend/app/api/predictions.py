"""API Routes for predictions."""
from fastapi import APIRouter, HTTPException, status
from datetime import datetime
from typing import List

from app.schemas.predictions import (
    FlowerFeatures,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo,
)
from app.services.predictor import get_predictor, PredictorService

router = APIRouter(prefix="/api/v1", tags=["predictions"])


@router.post("/predict", response_model=PredictionResponse)
async def predict(features: FlowerFeatures):
    """
    Predict flower species from input features.
    
    - **sepal_length**: Sepal length in cm
    - **sepal_width**: Sepal width in cm
    - **petal_length**: Petal length in cm
    - **petal_width**: Petal width in cm
    
    Returns the predicted species with confidence score.
    """
    predictor = get_predictor()
    
    if not predictor.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train the model first."
        )
    
    try:
        result = predictor.predict_single(
            sepal_length=features.sepal_length,
            sepal_width=features.sepal_width,
            petal_length=features.petal_length,
            petal_width=features.petal_width
        )
        
        return PredictionResponse(
            species=result["species"],
            species_id=result["species_id"],
            confidence=result["confidence"],
            probabilities=result.get("probabilities"),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch predict flower species for multiple samples.
    
    Accepts up to 100 samples in a single request.
    """
    predictor = get_predictor()
    
    if not predictor.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        predictions = []
        for features in request.samples:
            result = predictor.predict_single(
                sepal_length=features.sepal_length,
                sepal_width=features.sepal_width,
                petal_length=features.petal_length,
                petal_width=features.petal_width
            )
            predictions.append(PredictionResponse(
                species=result["species"],
                species_id=result["species_id"],
                confidence=result["confidence"],
                probabilities=result.get("probabilities"),
                timestamp=datetime.utcnow().isoformat()
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and model status."""
    predictor = get_predictor()
    return HealthResponse(
        status="healthy",
        model_loaded=predictor.is_model_loaded(),
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat()
    )


@router.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the loaded model."""
    predictor = get_predictor()
    info = predictor.get_model_info()
    
    if info is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    return ModelInfo(**info)


@router.get("/classes")
async def get_classes():
    """Get list of flower classes the model can predict."""
    predictor = get_predictor()
    classes = predictor.get_classes()
    
    return {
        "classes": classes,
        "count": len(classes)
    }
