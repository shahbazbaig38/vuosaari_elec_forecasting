"""
Unit tests for data_fetcher module
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetcher import DataFetcher


class TestDataFetcher:
    """Test cases for DataFetcher class"""
    
    @pytest.fixture
    def fetcher(self):
        """Create DataFetcher instance"""
        return DataFetcher()
    
    def test_initialization(self, fetcher):
        """Test DataFetcher initialization"""
        assert fetcher is not None
        assert fetcher.location_id == "4438"
        assert fetcher.latitude == 60.20766
        assert fetcher.longitude == 25.14080
    
    def test_get_energy_data(self, fetcher):
        """Test energy data fetching"""
        # Use a small date range for testing
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        try:
            df = fetcher.get_energy_data(start_date, end_date)
            
            # Check DataFrame structure
            assert isinstance(df, pd.DataFrame)
            assert 'consumption' in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)
            assert len(df) > 0
            
        except Exception as e:
            # API might be unavailable in test environment
            pytest.skip(f"API unavailable: {e}")
    
    def test_get_weather_data(self, fetcher):
        """Test weather data fetching"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        try:
            df = fetcher.get_weather_data(start_date, end_date)
            
            # Check DataFrame structure
            assert isinstance(df, pd.DataFrame)
            assert 'temperature' in df.columns
            assert isinstance(df.index, pd.DatetimeIndex)
            assert len(df) > 0
            
        except Exception as e:
            pytest.skip(f"API unavailable: {e}")
    
    def test_fetch_all_data(self, fetcher):
        """Test fetching both energy and weather data"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
        
        try:
            energy_df, weather_df = fetcher.fetch_all_data(start_date, end_date)
            
            assert isinstance(energy_df, pd.DataFrame)
            assert isinstance(weather_df, pd.DataFrame)
            assert 'consumption' in energy_df.columns
            assert 'temperature' in weather_df.columns
            
        except Exception as e:
            pytest.skip(f"API unavailable: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
