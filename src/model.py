"""
Model Training and Prediction Module

Handles model training, evaluation, saving, loading, and prediction
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from .config import get_config

logger = logging.getLogger(__name__)


class ElectricityForecastModel:
    """Machine learning model for electricity consumption forecasting"""
    
    def __init__(self, model_type: str = "RandomForestRegressor", **model_params):
        """
        Initialize model
        
        Args:
            model_type: Type of model to use
            **model_params: Model hyperparameters
        """
        self.config = get_config()
        self.model_type = model_type
        self.model_params = model_params or self.config.model_params
        self.model = None
        self.feature_names = None
        self.training_info = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on type"""
        if self.model_type == "RandomForestRegressor":
            self.model = RandomForestRegressor(**self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} with params: {self.model_params}")
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"🧠 Training {self.model_type}...")
        logger.info(f"  Training samples: {len(X_train)}")
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Train model
        start_time = datetime.now()
        self.model.fit(X_train, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"  ✅ Training completed in {training_time:.2f} seconds")
        
        # Evaluate on training set
        train_preds = self.model.predict(X_train)
        train_metrics = self._calculate_metrics(y_train, train_preds, "Training")
        
        # Evaluate on validation set if provided
        val_metrics = {}
        if X_val is not None and y_val is not None:
            logger.info(f"  Validation samples: {len(X_val)}")
            val_preds = self.model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_preds, "Validation")
        
        # Store training info
        self.training_info = {
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'n_train_samples': len(X_train),
            'training_time_seconds': training_time,
            'trained_at': datetime.now().isoformat(),
            'train_metrics': train_metrics,
            'val_metrics': val_metrics
        }
        
        return {**train_metrics, **val_metrics}
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, dataset_name: str) -> Dict[str, float]:
        """
        Calculate evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            dataset_name: Name of dataset (for logging)
            
        Returns:
            Dictionary of metrics
        """
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        logger.info(f"  📊 {dataset_name} Metrics:")
        logger.info(f"     MAE:  {mae:.2f} kWh")
        logger.info(f"     RMSE: {rmse:.2f} kWh")
        logger.info(f"     R²:   {r2:.4f}")
        logger.info(f"     MAPE: {mape:.2f}%")
        
        return {
            f'{dataset_name.lower()}_mae': mae,
            f'{dataset_name.lower()}_rmse': rmse,
            f'{dataset_name.lower()}_r2': r2,
            f'{dataset_name.lower()}_mape': mape
        }
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions
        
        Args:
            X: Features DataFrame
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        
        # Ensure features match training
        if self.feature_names:
            missing_features = set(self.feature_names) - set(X.columns)
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Reorder columns to match training
            X = X[self.feature_names]
        
        predictions = self.model.predict(X)
        
        return predictions
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance (for tree-based models)
        
        Returns:
            DataFrame with features and their importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        if not hasattr(self.model, 'feature_importances_'):
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save(self, filepath: Path = None):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save model. If None, uses config default.
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        if filepath is None:
            filepath = self.config.model_save_path
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'training_info': self.training_info
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"💾 Model saved to {filepath}")
    
    def load(self, filepath: Path = None):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to model file. If None, uses config default.
        """
        if filepath is None:
            filepath = self.config.model_save_path
        
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.feature_names = model_data.get('feature_names')
        self.training_info = model_data.get('training_info', {})
        
        logger.info(f"📂 Model loaded from {filepath}")
        logger.info(f"  Model type: {self.training_info.get('model_type', 'Unknown')}")
        logger.info(f"  Trained at: {self.training_info.get('trained_at', 'Unknown')}")
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
            Dictionary with model metadata
        """
        return self.training_info


def date_based_train_test_split(
    df: pd.DataFrame,
    train_end_date: str = None,
    test_size: int = 500
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data based on date
    
    Args:
        df: DataFrame with DatetimeIndex
        train_end_date: End date for training (YYYY-MM-DD). If None, uses last test_size rows for validation.
        test_size: Number of hours for validation if train_end_date is None
        
    Returns:
        Tuple of (train_df, test_df)
    """
    if train_end_date:
        train_end = pd.Timestamp(train_end_date)
        train_df = df[df.index <= train_end]
        test_df = df[df.index > train_end]
    else:
        # Use last test_size rows for validation
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]
    
    logger.info(f"Train/Test Split:")
    logger.info(f"  Training: {len(train_df)} samples ({train_df.index.min()} to {train_df.index.max()})")
    logger.info(f"  Testing:  {len(test_df)} samples ({test_df.index.min()} to {test_df.index.max()})")
    
    return train_df, test_df


def train_and_save_model(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    train_end_date: str = None,
    test_size: int = 500,
    save_path: Path = None
) -> ElectricityForecastModel:
    """
    Complete training pipeline
    
    Args:
        df: DataFrame with features and target
        features: List of feature column names
        target: Target column name
        train_end_date: End date for training (YYYY-MM-DD)
        test_size: Number of hours for validation
        save_path: Path to save model
        
    Returns:
        Trained model
    """
    logger.info("=" * 60)
    logger.info("  MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Split data
    train_df, test_df = date_based_train_test_split(df, train_end_date, test_size)
    
    # Prepare features and target
    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]
    
    # Initialize and train model
    config = get_config()
    model = ElectricityForecastModel(
        model_type=config.model_type,
        **config.model_params
    )
    
    metrics = model.train(X_train, y_train, X_test, y_test)
    
    # Display feature importance
    importance_df = model.get_feature_importance()
    if not importance_df.empty:
        logger.info("\n  🔝 Top 10 Important Features:")
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"     {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model.save(save_path)
    
    logger.info("=" * 60)
    logger.info("  ✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    
    return model
