"""
Unit tests for model module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import ElectricityForecastModel, date_based_train_test_split


class TestElectricityForecastModel:
    """Test cases for ElectricityForecastModel class"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='h')
        
        df = pd.DataFrame({
            'hour': dates.hour,
            'dayofweek': dates.dayofweek,
            'month': dates.month,
            'temperature': np.random.uniform(-10, 20, 1000),
            'lag_24h': np.random.uniform(100, 300, 1000),
            'consumption': np.random.uniform(100, 300, 1000)
        }, index=dates)
        
        return df
    
    @pytest.fixture
    def model(self):
        """Create model instance"""
        return ElectricityForecastModel(
            model_type="RandomForestRegressor",
            n_estimators=10,  # Small for testing
            random_state=42
        )
    
    def test_initialization(self, model):
        """Test model initialization"""
        assert model is not None
        assert model.model_type == "RandomForestRegressor"
        assert model.model is not None
    
    def test_train(self, model, sample_data):
        """Test model training"""
        features = ['hour', 'dayofweek', 'month', 'temperature', 'lag_24h']
        target = 'consumption'
        
        X_train = sample_data[features].iloc[:800]
        y_train = sample_data[target].iloc[:800]
        X_val = sample_data[features].iloc[800:]
        y_val = sample_data[target].iloc[800:]
        
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Check that metrics are returned
        assert 'training_mae' in metrics
        assert 'validation_mae' in metrics
        assert metrics['training_mae'] >= 0
        assert metrics['validation_mae'] >= 0
    
    def test_predict(self, model, sample_data):
        """Test model prediction"""
        features = ['hour', 'dayofweek', 'month', 'temperature', 'lag_24h']
        target = 'consumption'
        
        X_train = sample_data[features].iloc[:800]
        y_train = sample_data[target].iloc[:800]
        
        model.train(X_train, y_train)
        
        X_test = sample_data[features].iloc[800:810]
        predictions = model.predict(X_test)
        
        assert len(predictions) == 10
        assert all(predictions >= 0)
    
    def test_save_load(self, model, sample_data):
        """Test model saving and loading"""
        features = ['hour', 'dayofweek', 'month', 'temperature', 'lag_24h']
        target = 'consumption'
        
        X_train = sample_data[features].iloc[:800]
        y_train = sample_data[target].iloc[:800]
        
        model.train(X_train, y_train)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            model.save(tmp_path)
            
            # Load model
            new_model = ElectricityForecastModel()
            new_model.load(tmp_path)
            
            # Check loaded model
            assert new_model.feature_names == features
            assert new_model.model is not None
            
            # Test prediction with loaded model
            X_test = sample_data[features].iloc[800:810]
            predictions = new_model.predict(X_test)
            assert len(predictions) == 10
            
        finally:
            # Cleanup
            if tmp_path.exists():
                tmp_path.unlink()
    
    def test_feature_importance(self, model, sample_data):
        """Test feature importance extraction"""
        features = ['hour', 'dayofweek', 'month', 'temperature', 'lag_24h']
        target = 'consumption'
        
        X_train = sample_data[features].iloc[:800]
        y_train = sample_data[target].iloc[:800]
        
        model.train(X_train, y_train)
        
        importance_df = model.get_feature_importance()
        
        assert not importance_df.empty
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert len(importance_df) == len(features)


class TestTrainTestSplit:
    """Test train/test split function"""
    
    def test_date_based_split(self):
        """Test date-based train/test split"""
        dates = pd.date_range(start='2024-01-01', periods=1000, freq='h')
        df = pd.DataFrame({'value': range(1000)}, index=dates)
        
        train_df, test_df = date_based_train_test_split(df, test_size=200)
        
        assert len(train_df) == 800
        assert len(test_df) == 200
        assert train_df.index.max() < test_df.index.min()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
