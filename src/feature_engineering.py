"""
Feature Engineering Module

Creates features for the machine learning model
"""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta
from typing import List, Optional

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features from raw energy and weather data"""
    
    def __init__(self, lag_hours: List[int] = None, rolling_windows: List[int] = None):
        """
        Initialize feature engineer
        
        Args:
            lag_hours: List of lag periods in hours (e.g., [24, 48, 168])
            rolling_windows: List of rolling window sizes in hours (e.g., [24, 168])
        """
        self.lag_hours = lag_hours or [24]
        self.rolling_windows = rolling_windows or []
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp index
        
        Args:
            df: DataFrame with DatetimeIndex
            
        Returns:
            DataFrame with added temporal features
        """
        logger.info("Creating temporal features...")
        
        df = df.copy()
        
        # Basic temporal features
        df['hour'] = df.index.hour
        df['dayofweek'] = df.index.dayofweek
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofyear'] = df.index.dayofyear
        
        # Cyclical encoding for hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for day of week (7-day cycle)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        # Cyclical encoding for month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Weekend indicator
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
        
        # Season (meteorological)
        df['season'] = df['month'].apply(self._get_season)
        
        logger.info(f"  ✅ Added {len(['hour', 'dayofweek', 'month', 'day', 'dayofyear', 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'month_sin', 'month_cos', 'is_weekend', 'season'])} temporal features")
        
        return df
    
    @staticmethod
    def _get_season(month: int) -> int:
        """
        Get season from month (Northern Hemisphere)
        
        Args:
            month: Month number (1-12)
            
        Returns:
            Season code (0=Winter, 1=Spring, 2=Summer, 3=Autumn)
        """
        if month in [12, 1, 2]:
            return 0  # Winter
        elif month in [3, 4, 5]:
            return 1  # Spring
        elif month in [6, 7, 8]:
            return 2  # Summer
        else:
            return 3  # Autumn
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'consumption') -> pd.DataFrame:
        """
        Create lag features from target variable
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            DataFrame with added lag features
        """
        logger.info(f"Creating lag features for {target_col}...")
        
        df = df.copy()
        
        for lag in self.lag_hours:
            col_name = f'lag_{lag}h'
            df[col_name] = df[target_col].shift(lag)
            logger.info(f"  ✅ Added {col_name}")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'consumption') -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: DataFrame with target column
            target_col: Name of target column
            
        Returns:
            DataFrame with added rolling features
        """
        if not self.rolling_windows:
            return df
        
        logger.info(f"Creating rolling features for {target_col}...")
        
        df = df.copy()
        
        for window in self.rolling_windows:
            # Rolling mean
            df[f'rolling_mean_{window}h'] = df[target_col].rolling(window=window, min_periods=1).mean()
            
            # Rolling std
            df[f'rolling_std_{window}h'] = df[target_col].rolling(window=window, min_periods=1).std()
            
            # Rolling min/max
            df[f'rolling_min_{window}h'] = df[target_col].rolling(window=window, min_periods=1).min()
            df[f'rolling_max_{window}h'] = df[target_col].rolling(window=window, min_periods=1).max()
            
            logger.info(f"  ✅ Added rolling features for {window}h window")
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived weather features
        
        Args:
            df: DataFrame with temperature column
            
        Returns:
            DataFrame with added weather features
        """
        if 'temperature' not in df.columns:
            logger.warning("No temperature column found, skipping weather features")
            return df
        
        logger.info("Creating weather features...")
        
        df = df.copy()
        
        # Temperature squared (for non-linear effects)
        df['temperature_squared'] = df['temperature'] ** 2
        
        # Heating/cooling degree hours (base 18°C)
        df['heating_degree_hours'] = df['temperature'].apply(lambda x: max(0, 18 - x))
        df['cooling_degree_hours'] = df['temperature'].apply(lambda x: max(0, x - 18))
        
        # Temperature change from previous hour
        df['temp_change_1h'] = df['temperature'].diff()
        
        
        logger.info(f"  ✅ Added weather-derived features")
        
        return df
    
    def merge_energy_weather(self, energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge energy and weather data on timestamp
        
        Args:
            energy_df: Energy consumption DataFrame
            weather_df: Weather DataFrame
            
        Returns:
            Merged DataFrame
        """
        logger.info("Merging energy and weather data...")
        
        # Inner join to keep only timestamps with both energy and weather data
        df = energy_df.join(weather_df, how='inner')
        
        logger.info(f"  ✅ Merged data: {len(df)} rows")
        
        return df
    
    def prepare_training_data(self, energy_df: pd.DataFrame, weather_df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete feature engineering pipeline for training
        
        Args:
            energy_df: Energy consumption DataFrame
            weather_df: Weather DataFrame
            
        Returns:
            DataFrame with all features ready for training
        """
        logger.info("🔧 Starting feature engineering pipeline...")
        
        # Merge data
        df = self.merge_energy_weather(energy_df, weather_df)
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create weather features
        df = self.create_weather_features(df)
        
        # Create lag features
        df = self.create_lag_features(df, target_col='consumption')
        
        # Create rolling features
        df = self.create_rolling_features(df, target_col='consumption')
        
        # Drop rows with NaN values (created by lag/rolling features)
        initial_rows = len(df)
        df = df.dropna()
        dropped_rows = initial_rows - len(df)
        
        if dropped_rows > 0:
            logger.info(f"  ⚠️ Dropped {dropped_rows} rows with NaN values")
        
        logger.info(f"✅ Feature engineering complete: {len(df)} rows, {len(df.columns)} columns")
        
        return df
    
    def prepare_prediction_data(
        self, 
        future_dates: pd.DatetimeIndex,
        weather_df: pd.DataFrame,
        historical_consumption: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Prepare features for future predictions
        
        Args:
            future_dates: DatetimeIndex for future timestamps
            weather_df: Weather forecast DataFrame
            historical_consumption: Historical consumption for lag features
            
        Returns:
            DataFrame ready for prediction
        """
        logger.info("🔮 Preparing prediction features...")
        
        # Create future dataframe
        df_future = pd.DataFrame(index=future_dates)
        
        # Join with weather forecast
        df_future = df_future.join(weather_df)
        
        # Create temporal features
        df_future = self.create_temporal_features(df_future)
        
        # Create weather features
        df_future = self.create_weather_features(df_future)
        
        # Add lag features from historical data
        for lag in self.lag_hours:
            lag_values = []
            for date in df_future.index:
                past_time = date - timedelta(hours=lag)
                if past_time in historical_consumption.index:
                    val = historical_consumption.loc[past_time, 'consumption']
                else:
                    # Fallback: use mean of recent data
                    val = historical_consumption['consumption'].tail(168).mean()
                lag_values.append(val)
            
            df_future[f'lag_{lag}h'] = lag_values
        
        # Forward/backward fill any remaining NaN values
        df_future = df_future.ffill().bfill()
        
        logger.info(f"✅ Prediction features ready: {len(df_future)} rows")
        
        return df_future
