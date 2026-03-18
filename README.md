# Flower Classification - Production ML Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready, end-to-end machine learning project for flower species classification with REST API, CLI, and modern MLOps practices.

## 🌟 Features

- **Complete ML Pipeline**: Data loading → Preprocessing → Training → Evaluation → Deployment
- **Multiple Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **REST API**: FastAPI-based production API with Swagger documentation
- **CLI Tool**: Command-line interface for predictions
- **Docker Support**: Containerized deployment ready
- **Testing**: Unit and integration tests with pytest
- **Configuration**: YAML-based configuration management
- **Logging**: Structured logging with file and console output

## 📁 Project Structure

```
Flower-Classification/
├── backend/
│   ├── app/                    # FastAPI application
│   │   ├── api/                # API routes
│   │   ├── core/               # Core configuration
│   │   ├── models/             # Pydantic models
│   │   ├── schemas/            # Request/response schemas
│   │   ├── services/           # Business logic
│   │   └── utils/              # Utility functions
│   ├── src/                    # ML source code
│   │   ├── data/               # Data loading & preprocessing
│   │   ├── features/           # Feature engineering
│   │   ├── models/             # Model training
│   │   └── evaluation/         # Model evaluation
│   ├── scripts/                # CLI and training scripts
│   ├── tests/                  # Unit & integration tests
│   ├── notebooks/              # Jupyter notebooks
│   ├── data/                   # Data storage
│   ├── models/artifacts/       # Trained models
│   ├── configs/                # Configuration files
│   ├── docker/                 # Docker files
│   └── logs/                   # Application logs
├── pyproject.toml              # Project configuration
├── requirements.txt            # Python dependencies
├── Makefile                    # Common commands
└── README.md                   # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or poetry

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/flower-classification.git
cd flower-classification

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Or install as package (development mode)
pip install -e ".[dev]"
```

### Training the Model

```bash
# Using Makefile
make train

# Or directly
python backend/scripts/train.py --sample

# With custom data
python backend/scripts/train.py --data path/to/data.csv
```

### Running the API

```bash
# Using Makefile
make api

# Or directly
python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for interactive API documentation.

### Using the CLI

```bash
# Single prediction
python backend/scripts/cli.py predict --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2 -v

# Demo prediction
python backend/scripts/cli.py demo

# Model info
python backend/scripts/cli.py info
```

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API information |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI documentation |
| POST | `/api/v1/predict` | Single prediction |
| POST | `/api/v1/predict/batch` | Batch predictions |
| GET | `/api/v1/classes` | List available classes |
| GET | `/api/v1/model/info` | Model information |

### Example API Request

```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```

Response:
```json
{
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
```

## 🐳 Docker Deployment

```bash
# Build Docker image
make docker-build

# Run container
make docker-run

# Or with docker-compose (includes Jupyter)
make docker-compose-up
```

## 🧪 Testing

```bash
# Run all tests
make test

# Run with coverage
make test-cov

# Run specific test file
pytest backend/tests/unit/test_preprocessor.py -v
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.97 | ~0.97 | ~0.97 | ~0.97 |
| Decision Tree | ~0.95 | ~0.95 | ~0.95 | ~0.95 |
| Random Forest | ~0.98 | ~0.98 | ~0.98 | ~0.98 |
| XGBoost | ~0.98 | ~0.98 | ~0.98 | ~0.98 |

## 📝 Configuration

Edit `backend/configs/config.yaml` to customize:

- Data paths and split ratios
- Model hyperparameters
- API settings
- Logging configuration

## 🔧 Development

```bash
# Install development dependencies
make dev

# Format code
make format

# Run linters
make lint

# Run all checks
make check
```

## 📦 Using Your Own Dataset

Create a CSV file with columns:
```csv
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,0
...
```

Train with:
```bash
python backend/scripts/train.py --data path/to/your/data.csv
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Iris Flower Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- Framework: [FastAPI](https://fastapi.tiangolo.com/)
- ML Library: [scikit-learn](https://scikit-learn.org/)
