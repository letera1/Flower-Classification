"""Flower Classification API - FastAPI Application."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging

from app.core.config import get_settings
from app.api.predictions import router as predictions_router
from app.services.predictor import get_predictor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Flower Classification API...")
    settings = get_settings()
    logger.info(f"Models directory: {settings.MODELS_DIR}")
    
    # Preload model
    predictor = get_predictor()
    if predictor.is_model_loaded():
        logger.info("Model loaded successfully")
    else:
        logger.warning("Model not loaded - API will run in limited mode")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Flower Classification API...")


# Create FastAPI application
settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": settings.APP_DESCRIPTION,
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict"
    }


@app.get("/health", tags=["health"])
async def health():
    """Simple health check endpoint."""
    return {"status": "healthy"}


# Include routers
app.include_router(predictions_router)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__
        }
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD
    )
