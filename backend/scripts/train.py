#!/usr/bin/env python3
"""Training pipeline for flower classification."""
import sys
import json
import logging
import argparse
from pathlib import Path

# Add backend to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.data.preprocessor import DataProcessor
from src.features.engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.evaluation.evaluator import ModelEvaluator
from app.core.config import get_settings, ConfigLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_pipeline(data_path: str = None, use_sample: bool = True, tune: bool = True):
    """Run the complete training pipeline."""
    logger.info("=" * 60)
    logger.info("FLOWER CLASSIFICATION - TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Load configuration
    config = ConfigLoader().load()
    settings = get_settings()
    
    # 1. Load and preprocess data
    logger.info("\n[1/5] Loading and preprocessing data...")
    processor = DataProcessor()
    
    if use_sample or data_path is None:
        df = processor.load_sample_data()
    else:
        df = processor.load_data(data_path)
    
    df = processor.clean_data()
    X, y = processor.get_features_and_target(target_col='species')
    X_train, X_test, y_train, y_test = processor.split_data(
        X, y, 
        test_size=config['data']['test_size'],
        random_state=config['data']['random_state']
    )
    
    X_train_scaled = processor.fit_transform(X_train)
    X_test_scaled = processor.transform(X_test)
    
    logger.info(f'Feature columns: {processor.feature_columns}')
    logger.info(f'Training samples: {len(X_train)}, Test samples: {len(X_test)}')
    
    # 2. Initialize and train models
    logger.info("\n[2/5] Training models...")
    trainer = ModelTrainer(random_state=config['data']['random_state'])
    trainer.initialize_models()
    trainer.train_all(X_train_scaled, y_train)
    
    # 3. Hyperparameter tuning
    if tune:
        logger.info("\n[3/5] Hyperparameter tuning...")
        primary_model = config['models']['primary']
        trainer.tune_hyperparameters(
            primary_model, 
            X_train_scaled, 
            y_train, 
            cv=config['tuning']['cv_folds']
        )
    
    # 4. Evaluate all models
    logger.info("\n[4/5] Evaluating models...")
    class_names = ['setosa', 'versicolor', 'virginica']
    evaluator = ModelEvaluator(class_names=class_names)
    
    models_metrics = {}
    for name, model in trainer.models.items():
        y_pred = model.predict(X_test_scaled)
        metrics = evaluator.compute_metrics(y_test, y_pred)
        models_metrics[name] = metrics
        logger.info(f'{name}: Accuracy={metrics["accuracy"]:.4f}, F1={metrics["f1"]:.4f}')
    
    # Find best model
    best_model_name = max(models_metrics, key=lambda x: models_metrics[x]['accuracy'])
    best_model = trainer.models[best_model_name]
    logger.info(f'\nBest model: {best_model_name} with accuracy {models_metrics[best_model_name]["accuracy"]:.4f}')
    
    # 5. Save models and artifacts
    logger.info("\n[5/5] Saving models and artifacts...")
    models_dir = settings.MODELS_DIR
    plots_dir = ROOT / "notebooks" / "plots"
    models_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Save best model and preprocessor
    trainer.save_model(best_model_name, str(models_dir / 'best_model.joblib'))
    import joblib
    joblib.dump(processor.scaler, str(models_dir / 'preprocessor.joblib'))
    
    # Save metadata
    metadata = {
        'model_name': best_model_name,
        'class_names': class_names,
        'feature_columns': processor.feature_columns,
        'metrics': models_metrics[best_model_name],
        'training_samples': int(len(X_train)),
        'test_samples': int(len(X_test)),
        'all_metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in models_metrics.items()}
    }
    with open(models_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate evaluation plots
    y_pred_best = best_model.predict(X_test_scaled)
    evaluator.plot_confusion_matrix(
        y_test, y_pred_best,
        save_path=str(plots_dir / 'confusion_matrix.png')
    )
    evaluator.plot_metrics_comparison(
        models_metrics,
        save_path=str(plots_dir / 'metrics_comparison.png')
    )
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)
    logger.info(f'Models saved to: {models_dir}')
    logger.info(f'Plots saved to: {plots_dir}')
    
    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'metrics': models_metrics,
        'processor': processor
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train flower classification model')
    parser.add_argument('--data', type=str, help='Path to training data CSV')
    parser.add_argument('--sample', action='store_true', help='Use sample Iris data')
    parser.add_argument('--no-tune', action='store_true', help='Skip hyperparameter tuning')
    args = parser.parse_args()
    
    run_pipeline(
        data_path=args.data, 
        use_sample=args.sample or args.data is None,
        tune=not args.no_tune
    )
