"""Feature engineering module."""
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Create and transform features for flower classification."""
    
    def __init__(self):
        self.feature_names = []
        
    def create_ratio_features(self, df: pd.DataFrame, 
                               col1: str, col2: str,
                               new_col_name: str = None) -> pd.DataFrame:
        """Create ratio feature between two columns."""
        if new_col_name is None:
            new_col_name = f'{col1}_to_{col2}_ratio'
        df[new_col_name] = df[col1] / (df[col2] + 1e-6)
        logger.info(f'Created ratio feature: {new_col_name}')
        return df
    
    def create_size_features(self, df: pd.DataFrame,
                            length_col: str, width_col: str) -> pd.DataFrame:
        """Create size-related features (area, perimeter approximation)."""
        area_col = 'area'
        df[area_col] = df[length_col] * df[width_col]
        perimeter_col = 'perimeter'
        df[perimeter_col] = 2 * (df[length_col] + df[width_col])
        logger.info(f'Created size features: {area_col}, {perimeter_col}')
        return df
    
    def create_interaction_features(self, df: pd.DataFrame,
                                    columns: list) -> pd.DataFrame:
        """Create interaction features between multiple columns."""
        for i, col1 in enumerate(columns):
            for col2 in columns[i+1:]:
                new_col = f'{col1}_x_{col2}'
                df[new_col] = df[col1] * df[col2]
        logger.info('Created interaction features')
        return df
    
    def get_feature_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for all features."""
        numeric_df = df.select_dtypes(include=[np.number])
        summary = numeric_df.describe().T
        summary['skewness'] = numeric_df.skew()
        summary['kurtosis'] = numeric_df.kurtosis()
        return summary
