"""Integration tests for API endpoints."""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
def client():
    """Create test client."""
    from backend.app.main import app
    return TestClient(app)


class TestAPIEndpoints:
    """Tests for API endpoints."""
    
    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
    
    def test_health(self, client):
        """Test health endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_predict_endpoint_structure(self, client):
        """Test predict endpoint accepts valid input."""
        payload = {
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/api/v1/predict", json=payload)
        
        # May return 503 if model not loaded, but should accept valid input
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "species" in data
            assert "confidence" in data
    
    def test_predict_invalid_input(self, client):
        """Test predict endpoint rejects invalid input."""
        payload = {
            "sepal_length": -1,  # Invalid negative value
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
        
        response = client.post("/api/v1/predict", json=payload)
        
        assert response.status_code == 422  # Validation error
    
    def test_classes_endpoint(self, client):
        """Test classes endpoint."""
        response = client.get("/api/v1/classes")
        
        assert response.status_code == 200
        data = response.json()
        assert "classes" in data
        assert "count" in data
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get("/api/v1/model/info")
        
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]


class TestBatchPrediction:
    """Tests for batch prediction endpoint."""
    
    def test_batch_predict(self, client):
        """Test batch prediction with multiple samples."""
        payload = {
            "samples": [
                {
                    "sepal_length": 5.1,
                    "sepal_width": 3.5,
                    "petal_length": 1.4,
                    "petal_width": 0.2
                },
                {
                    "sepal_length": 6.2,
                    "sepal_width": 2.9,
                    "petal_length": 4.3,
                    "petal_width": 1.3
                }
            ]
        }
        
        response = client.post("/api/v1/predict/batch", json=payload)
        
        # May return 503 if model not loaded
        assert response.status_code in [200, 503]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert len(data["predictions"]) == 2
