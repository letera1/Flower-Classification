"""Data loading and preprocessing module."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handle data loading, cleaning, and preprocessing."""
    
    def __init__(self, raw_data_path: str = None):
        self.raw_data_path = raw_data_path
        self.df = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self, filepath: str = None) -> pd.DataFrame:
        """Load data from CSV file."""
        path = filepath or self.raw_data_path
        if path is None:
            raise ValueError('No data path provided')
        
        logger.info(f'Loading data from {path}')
        self.df = pd.read_csv(path)
        logger.info(f'Data shape: {self.df.shape}')
        return self.df
    
    def load_sample_data(self) -> pd.DataFrame:
        """Load sample flower dataset for demonstration."""
        from sklearn.datasets import load_iris
        
        logger.info('Loading sample Iris dataset')
        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.df['species'] = iris.target
        
        species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        self.df['species_name'] = self.df['species'].map(species_map)
        
        logger.info(f'Data shape: {self.df.shape}')
        return self.df
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset."""
        if self.df is None:
            raise ValueError('No data loaded')
        
        logger.info('Cleaning data')
        before = len(self.df)
        self.df = self.df.drop_duplicates()
        logger.info(f'Dropped {before - len(self.df)} duplicates')
        self.df = self.df.dropna()
        logger.info('Dropped rows with missing values')
        return self.df
    
    def get_features_and_target(self, target_col: str = 'species') -> tuple:
        """Separate features and target."""
        if self.df is None:
            raise ValueError('No data loaded')
        
        self.feature_columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in self.feature_columns:
            self.feature_columns.remove(target_col)
        
        X = self.df[self.feature_columns]
        y = self.df[target_col]
        
        logger.info(f'Features: {self.feature_columns}')
        return X, y
    
    def split_data(self, X: pd.DataFrame, y: pd.Series, 
                   test_size: float = 0.2, random_state: int = 42) -> tuple:
        """Split data into train and test sets."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f'Train size: {len(X_train)}, Test size: {len(X_test)}')
        return X_train, X_test, y_train, y_test
    
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler and transform features."""
        X_scaled = self.scaler.fit_transform(X)
        logger.info('Fitted and transformed features')
        return X_scaled
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features using fitted scaler."""
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target labels."""
        y_encoded = self.label_encoder.fit_transform(y)
        logger.info(f'Encoded classes: {self.label_encoder.classes_}')
        return y_encoded
    
    def decode_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Decode encoded labels back to original."""
        return self.label_encoder.inverse_transform(y_encoded)
