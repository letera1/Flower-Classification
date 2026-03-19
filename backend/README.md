# Flower Classification - Jupyter Notebook ML Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A **Jupyter notebook-based** machine learning pipeline for flower species classification. This project provides an interactive, educational approach to ML with comprehensive visualizations and step-by-step analysis.

## 🌟 Features

- **Interactive Notebooks**: Step-by-step ML pipeline with visualizations
- **Complete Workflow**: EDA → Preprocessing → Feature Engineering → Training → Evaluation
- **Multiple Models**: Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Rich Visualizations**: Confusion matrices, feature importance, metrics comparison
- **Reproducible**: Save/load models and preprocessed data
- **Educational**: Well-documented code with explanations

## 📊 Notebook Pipeline

| Notebook | Description |
|----------|-------------|
| **00_master_pipeline.ipynb** | 🎯 Complete end-to-end pipeline (quick start) |
| **01_eda_flower_classification.ipynb** | 📈 Exploratory Data Analysis |
| **02_data_preprocessing.ipynb** | 🧹 Data loading, cleaning, scaling, splitting |
| **03_feature_engineering.ipynb** | 🔧 Creating new features (ratios, interactions, polynomials) |
| **04_model_training.ipynb** | 🤖 Training & comparing multiple models |
| **05_model_evaluation.ipynb** | 📊 Comprehensive evaluation & visualization |

## 📁 Project Structure

```
backend/
├── notebooks/                    # Jupyter notebooks (main workflow)
│   ├── 00_master_pipeline.ipynb
│   ├── 01_eda_flower_classification.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   └── 05_model_evaluation.ipynb
├── app/                          # FastAPI application (optional deployment)
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   └── utils/
├── data/
│   ├── raw/                      # Raw data files
│   └── processed/                # Preprocessed data (auto-generated)
├── models/
│   └── artifacts/                # Trained models (auto-generated)
├── configs/                      # Configuration files
├── scripts/                      # Utility scripts
├── tests/                        # Unit tests
├── docker/                       # Docker configuration
├── logs/                         # Application logs
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- Jupyter Notebook

### Installation

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Running the Pipeline

**Option 1: Quick Start (Recommended)**
```bash
# Open the master pipeline notebook
jupyter notebook notebooks/00_master_pipeline.ipynb
```

**Option 2: Step-by-Step**
```bash
# Start Jupyter
jupyter notebook

# Then open notebooks in order:
# 1. 01_eda_flower_classification.ipynb
# 2. 02_data_preprocessing.ipynb
# 3. 03_feature_engineering.ipynb
# 4. 04_model_training.ipynb
# 5. 05_model_evaluation.ipynb
```

## 📖 Notebook Guide

### 1. EDA (01_eda_flower_classification.ipynb)
- Load and explore the Iris dataset
- Visualize feature distributions
- Analyze class balance
- Check correlations

### 2. Data Preprocessing (02_data_preprocessing.ipynb)
- Clean data (handle missing values, duplicates)
- Split into train/test sets
- Feature scaling with StandardScaler
- Target encoding with LabelEncoder
- Save preprocessed data

### 3. Feature Engineering (03_feature_engineering.ipynb)
- Create ratio features
- Generate size features (area, perimeter)
- Build interaction features
- Add polynomial features
- Analyze feature correlations

### 4. Model Training (04_model_training.ipynb)
- Initialize 4 classification models
- Train all models
- Cross-validation (5-fold)
- Hyperparameter tuning
- Feature importance analysis
- Save trained models

### 5. Model Evaluation (05_model_evaluation.ipynb)
- Compute metrics (accuracy, precision, recall, F1)
- Generate confusion matrices
- Analyze misclassifications
- Compare all models
- Save evaluation results

## 📈 Expected Results

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.97 | ~0.97 | ~0.97 | ~0.97 |
| Decision Tree | ~0.95 | ~0.95 | ~0.95 | ~0.95 |
| Random Forest | ~0.98 | ~0.98 | ~0.98 | ~0.98 |
| XGBoost | ~0.98 | ~0.98 | ~0.98 | ~0.98 |

## 🎯 Using the Trained Model

After running the pipeline, models are saved in `models/artifacts/`:

```python
import joblib
from pathlib import Path
import numpy as np

# Load the best model
model = joblib.load('models/artifacts/best_model.joblib')
scaler = joblib.load('models/artifacts/scaler.joblib')
label_encoder = joblib.load('models/artifacts/label_encoder.joblib')

# Make a prediction
sample = np.array([[5.1, 3.5, 1.4, 0.2]])  # sepal L/W, petal L/W
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

# Decode the result
species = label_encoder.inverse_transform(prediction)
print(f'Predicted species: {species[0]}')
```

## 🐳 Docker (Optional)

For containerized deployment:

```bash
# Build Docker image
docker build -f docker/Dockerfile -t flower-classification .

# Run with Jupyter
docker run -p 8888:8888 -v $(pwd):/app flower-classification
```

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

## 📝 Configuration

Key parameters you can modify in the notebooks:

```python
RANDOM_STATE = 42    # For reproducibility
TEST_SIZE = 0.2      # Test set ratio
CV_FOLDS = 5         # Cross-validation folds
```

## 🔧 Development

```bash
# Install development dependencies
pip install pytest black flake8

# Format code
black notebooks/*.ipynb

# Run linter
flake8 scripts/
```

## 📦 Generated Artifacts

After running the pipeline:

```
models/artifacts/
├── best_model.joblib           # Best performing model
├── best_model_info.joblib      # Model info and metrics
├── scaler.joblib               # Fitted StandardScaler
├── label_encoder.joblib        # Fitted LabelEncoder
├── feature_columns.joblib      # Feature names
├── evaluation_metrics.csv      # All model metrics
└── model_comparison_results.csv # Training results

data/processed/
├── X_train.npy                 # Training features
├── X_test.npy                  # Test features
├── y_train.npy                 # Training labels
├── y_test.npy                  # Test labels
└── data_with_engineered_features.csv  # Full dataset
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes in notebooks
4. Submit a Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset: [Iris Flower Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- Tools: [Jupyter](https://jupyter.org/), [scikit-learn](https://scikit-learn.org/)
- Visualization: [matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/)

## 📞 Support

For issues or questions, please open an issue on GitHub.
