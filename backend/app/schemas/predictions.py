"""Pydantic schemas for API request/response validation."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class FlowerFeatures(BaseModel):
    """Input features for flower classification."""
    
    sepal_length: float = Field(
        ..., 
        description="Sepal length in centimeters",
        ge=0.0,
        le=10.0,
        examples=[5.1]
    )
    sepal_width: float = Field(
        ..., 
        description="Sepal width in centimeters",
        ge=0.0,
        le=5.0,
        examples=[3.5]
    )
    petal_length: float = Field(
        ..., 
        description="Petal length in centimeters",
        ge=0.0,
        le=10.0,
        examples=[1.4]
    )
    petal_width: float = Field(
        ..., 
        description="Petal width in centimeters",
        ge=0.0,
        le=3.0,
        examples=[0.2]
    )
    
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
    """Response from single prediction endpoint."""
    
    species: str = Field(..., description="Predicted flower species")
    species_id: int = Field(..., description="Numeric ID of predicted species")
    confidence: float = Field(..., description="Prediction confidence (0-1)")
    probabilities: Optional[Dict[str, float]] = Field(
        None, 
        description="Class probabilities"
    )
    timestamp: str = Field(..., description="Prediction timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "species": "setosa",
                "species_id": 0,
                "confidence": 0.98,
                "probabilities": {
                    "setosa": 0.98,
                    "versicolor": 0.01,
                    "virginica": 0.01
                },
                "timestamp": "2024-01-01T00:00:00"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    
    samples: List[FlowerFeatures] = Field(
        ..., 
        description="List of flower samples to classify",
        min_length=1,
        max_length=100
    )


class BatchPredictionResponse(BaseModel):
    """Response from batch prediction endpoint."""
    
    predictions: List[PredictionResponse]
    total: int
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Response timestamp")


class ModelInfo(BaseModel):
    """Model information response."""
    
    model_name: str
    classes: List[str]
    features: List[str]
    accuracy: Optional[float] = None
    training_samples: Optional[int] = None
    test_samples: Optional[int] = None


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    detail: str
    error_type: str
    timestamp: str
