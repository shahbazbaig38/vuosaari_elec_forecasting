"""
Unit tests for feature_engineering module
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class"""
    
    @pytest.fixture
    def engineer(self):
        """Create FeatureEngineer instance"""
        return FeatureEngineer(lag_hours=[24], rolling_windows=[])
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
        df = pd.DataFrame({
            'consumption': np.random.uniform(100, 300, 100),
            'temperature': np.random.uniform(-10, 20, 100)
        }, index=dates)
        return df
    
    def test_initialization(self, engineer):
        """Test FeatureEngineer initialization"""
        assert engineer is not None
        assert engineer.lag_hours == [24]
        assert engineer.rolling_windows == []
    
    def test_create_temporal_features(self, engineer, sample_data):
        """Test temporal feature creation"""
        df = engineer.create_temporal_features(sample_data)
        
        # Check that temporal features are created
        assert 'hour' in df.columns
        assert 'dayofweek' in df.columns
        assert 'month' in df.columns
        assert 'is_weekend' in df.columns
        assert 'season' in df.columns
        
        # Check value ranges
        assert df['hour'].min() >= 0
        assert df['hour'].max() <= 23
        assert df['dayofweek'].min() >= 0
        assert df['dayofweek'].max() <= 6
    
    def test_create_lag_features(self, engineer, sample_data):
        """Test lag feature creation"""
        df = engineer.create_lag_features(sample_data, target_col='consumption')
        
        # Check lag feature exists
        assert 'lag_24h' in df.columns
        
        # Check lag values (first 24 should be NaN)
        assert df['lag_24h'].iloc[:24].isna().all()
        assert not df['lag_24h'].iloc[24:].isna().all()
    
    def test_create_weather_features(self, engineer, sample_data):
        """Test weather feature creation"""
        df = engineer.create_weather_features(sample_data)
        
        # Check weather features
        assert 'temperature_squared' in df.columns
        assert 'heating_degree_hours' in df.columns
        assert 'cooling_degree_hours' in df.columns
    
    def test_merge_energy_weather(self, engineer):
        """Test merging energy and weather data"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='h')
        
        energy_df = pd.DataFrame({
            'consumption': np.random.uniform(100, 300, 50)
        }, index=dates)
        
        weather_df = pd.DataFrame({
            'temperature': np.random.uniform(-10, 20, 50)
        }, index=dates)
        
        merged_df = engineer.merge_energy_weather(energy_df, weather_df)
        
        assert 'consumption' in merged_df.columns
        assert 'temperature' in merged_df.columns
        assert len(merged_df) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
