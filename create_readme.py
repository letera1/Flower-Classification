#!/usr/bin/env python3
"""Create README file."""

readme = """# Flower Classification - Machine Learning Project

A production-ready end-to-end machine learning project for flower species classification using morphological features.

## 🌸 Project Overview

This project demonstrates a complete ML pipeline for classifying flower species (Iris dataset) with:
- **Exploratory Data Analysis (EDA)** - Understanding data patterns and relationships
- **Multiple ML Models** - Logistic Regression, Decision Tree, Random Forest, XGBoost
- **Hyperparameter Tuning** - Grid search optimization
- **Comprehensive Evaluation** - Metrics, confusion matrices, error analysis
- **REST API Backend** - FastAPI-based prediction service
- **CLI Tool** - Command-line interface for predictions

## 📁 Project Structure

```
Flower-Classification/
├── backend/
│   └── app.py              # FastAPI application
├── notebooks/
│   └── 01_eda_flower_classification.ipynb
├── src/
│   ├── data/
│   │   ├── preprocessor.py      # Data loading & preprocessing
│   │   └── feature_engineering.py
│   ├── models/
│   │   └── trainer.py           # Model training & tuning
│   ├── evaluation/
│   │   └── evaluator.py         # Metrics & visualization
│   └── utils/
│       └── helpers.py           # Utility functions
├── cli/
│   └── predict.py          # CLI tool for predictions
├── data/
│   ├── raw/                # Raw data files
│   └── processed/          # Processed data files
├── models/                 # Trained model artifacts
├── train_pipeline.py       # Main training script
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\\Scripts\\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# Train with sample Iris data
python train_pipeline.py --sample

# Train with custom dataset
python train_pipeline.py --data path/to/data.csv
```

### Using the CLI

```bash
# Single prediction
python -m cli.predict predict --sepal-length 5.1 --sepal-width 3.5 --petal-length 1.4 --petal-width 0.2 -v

# Batch prediction from CSV
python -m cli.predict predict-file --file input.csv

# Model information
python -m cli.predict info

# Demo prediction
python -m cli.predict demo
```

### Running the API

```bash
# Start the FastAPI server
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# Or directly
cd backend && python app.py
```

API endpoints:
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - Interactive API documentation (Swagger UI)
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /classes` - List available classes

### API Usage Example

```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \\
  -H "Content-Type: application/json" \\
  -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \\
  -H "Content-Type: application/json" \\
  -d '[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]'
```

## 📊 Dataset

This project uses the **Iris Flower Dataset**:
- **Samples**: 150 flowers (50 per species)
- **Features**: 
  - Sepal length (cm)
  - Sepal width (cm)
  - Petal length (cm)
  - Petal width (cm)
- **Classes**: setosa, versicolor, virginica

### Using Custom Data

For custom flower datasets, create a CSV with columns:
```
sepal_length,sepal_width,petal_length,petal_width,species
5.1,3.5,1.4,0.2,0
...
```

## 📈 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~0.97 | ~0.97 | ~0.97 | ~0.97 |
| Decision Tree | ~0.95 | ~0.95 | ~0.95 | ~0.95 |
| Random Forest | ~0.98 | ~0.98 | ~0.98 | ~0.98 |
| XGBoost | ~0.98 | ~0.98 | ~0.98 | ~0.98 |

*Actual performance may vary based on train/test split*

## 🔍 Key Features

### Data Processing
- Automated data loading and cleaning
- Feature scaling with StandardScaler
- Label encoding for target variable
- Train/test split with stratification

### Model Training
- 4 classification algorithms
- Grid search hyperparameter tuning
- Cross-validation support
- Model persistence with joblib

### Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix visualization
- Misclassification analysis
- ROC curves for multi-class

## 🛠️ Technologies

- **Python 3.9+**
- **scikit-learn** - Machine learning
- **XGBoost** - Gradient boosting
- **FastAPI** - REST API
- **pandas, numpy** - Data processing
- **matplotlib, seaborn** - Visualization
- **click** - CLI framework

## 📝 License

MIT License - See LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue on GitHub.
"""

with open('README.md', 'w', encoding='utf-8') as f:
    f.write(readme)
print('Created README.md')
