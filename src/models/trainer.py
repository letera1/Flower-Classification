"""Model training and management module."""
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import xgboost as xgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage multiple classification models."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all models with default parameters."""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                multi_class='multinomial'
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state
            ),
            'random_forest': RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        }
        logger.info(f'Initialized {len(self.models)} models')
        return self.models
    
    def train_model(self, name: str, X_train: np.ndarray, 
                    y_train: np.ndarray) -> Any:
        """Train a specific model."""
        if name not in self.models:
            raise ValueError(f'Model {name} not found')
        
        logger.info(f'Training {name}...')
        model = self.models[name]
        model.fit(X_train, y_train)
        logger.info(f'{name} training completed')
        return model
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """Train all initialized models."""
        for name in self.models:
            self.train_model(name, X_train, y_train)
        return self.models
    
    def tune_hyperparameters(self, model_name: str, X_train: np.ndarray,
                            y_train: np.ndarray, cv: int = 5) -> Dict:
        """Perform hyperparameter tuning for a specific model."""
        param_grids = {
            'logistic_regression': {
                'C': [0.01, 0.1, 1, 10, 100],
                'solver': ['lbfgs', 'saga'],
                'penalty': ['l2']
            },
            'decision_tree': {
                'max_depth': [3, 5, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        }
        
        if model_name not in param_grids:
            raise ValueError(f'No param grid for {model_name}')
        
        logger.info(f'Tuning hyperparameters for {model_name}...')
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        logger.info(f'Best params for {model_name}: {grid_search.best_params_}')
        logger.info(f'Best CV score for {model_name}: {grid_search.best_score_:.4f}')
        
        self.models[model_name] = grid_search.best_estimator_
        return {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_
        }
    
    def cross_validate(self, model_name: str, X: np.ndarray, 
                       y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation on a model."""
        if model_name not in self.models:
            raise ValueError(f'Model {model_name} not found')
        
        scores = cross_val_score(
            self.models[model_name], X, y, cv=cv, scoring='accuracy'
        )
        return {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
    
    def save_model(self, model_name: str, save_path: str) -> str:
        """Save a trained model to disk."""
        if model_name not in self.models:
            raise ValueError(f'Model {model_name} not found')
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.models[model_name], save_path)
        logger.info(f'Model saved to {save_path}')
        return save_path
    
    def load_model(self, model_path: str) -> Any:
        """Load a trained model from disk."""
        model = joblib.load(model_path)
        logger.info(f'Model loaded from {model_path}')
        return model
    
    def get_feature_importance(self, model_name: str) -> Optional[np.ndarray]:
        """Get feature importance from tree-based models."""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        if hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        return None
