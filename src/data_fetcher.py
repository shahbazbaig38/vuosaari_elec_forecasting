"""
Data Fetching Module

Handles fetching electricity consumption and weather data from external APIs
"""

import requests
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Tuple, Optional
from .config import get_config

logger = logging.getLogger(__name__)


class DataFetcher:
    """Fetches data from Nuuka and Open-Meteo APIs"""
    
    def __init__(self):
        """Initialize data fetcher with configuration"""
        self.config = get_config()
        self.nuuka_base_url = self.config.nuuka_base_url
        self.location_id = self.config.location_id
        self.latitude = self.config.latitude
        self.longitude = self.config.longitude
        self.timezone = self.config.timezone
    
    def get_energy_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch electricity consumption from Nuuka API
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with timestamp index and consumption column
            
        Raises:
            Exception: If API request fails or returns no data
        """
        params = {
            "Record": "LocationName",
            "SearchString": self.location_id,
            "ReportingGroup": "Electricity",
            "StartTime": start_date,
            "EndTime": end_date
        }
        
        logger.info(f"🔌 Fetching Energy Data ({start_date} to {end_date})...")
        
        try:
            response = requests.get(self.nuuka_base_url, params=params, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Nuuka API request failed: {e}")
            raise Exception(f"Nuuka API Error: {e}")
        
        data = response.json()
        
        if not data:
            logger.warning("No energy data found for this range")
            raise Exception("No energy data found for this range.")
        
        df = pd.DataFrame(data)
        
        # Standardize columns based on API response structure
        if 'reportingTime' in df.columns:
            df['timestamp'] = pd.to_datetime(df['reportingTime'])
        elif 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        else:
            logger.error("No timestamp column found in API response")
            raise Exception("Invalid API response format")
        
        df['consumption'] = pd.to_numeric(df['value'], errors='coerce')
        
        # Set index and sort
        df = df.set_index('timestamp').sort_index()
        
        # Remove duplicates and resample to hourly
        df = df[~df.index.duplicated(keep='first')]
        df = df[['consumption']].resample('h').mean()
        
        logger.info(f"✅ Fetched {len(df)} hours of energy data")
        
        return df
    
    def get_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch temperature data from Open-Meteo API
        Combines historical (archive) and forecast data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            DataFrame with timestamp index and temperature column
            
        Raises:
            Exception: If API request fails
        """
        logger.info(f"☁️ Fetching Weather Data ({start_date} to {end_date})...")
        
        # Determine split point between archive and forecast
        # Archive API: historical data up to ~5 days ago
        # Forecast API: recent past + future
        cutoff_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
        
        df_hist = pd.DataFrame()
        df_fore = pd.DataFrame()
        
        # Fetch historical data if needed
        if start_date < cutoff_date:
            df_hist = self._fetch_archive_weather(start_date, cutoff_date)
        
        # Fetch forecast data (includes recent past and future)
        df_fore = self._fetch_forecast_weather()
        
        # Combine and deduplicate
        df_weather = pd.concat([df_hist, df_fore])
        df_weather = df_weather[~df_weather.index.duplicated(keep='last')]
        df_weather = df_weather.sort_index()
        
        # Resample to hourly and interpolate missing values
        df_weather = df_weather.resample('h').mean().interpolate(method='linear')
        
        logger.info(f"✅ Fetched {len(df_weather)} hours of weather data")
        
        return df_weather
    
    def _fetch_archive_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical weather from Archive API"""
        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": "temperature_2m",
            "timezone": self.timezone
        }
        
        try:
            response = requests.get(archive_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather Archive API request failed: {e}")
            raise Exception(f"Weather Archive API Error: {e}")
        
        if 'hourly' not in data:
            logger.warning("No historical weather data available")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['hourly']['time']),
            'temperature': data['hourly']['temperature_2m']
        })
        
        df = df.set_index('timestamp')
        logger.info(f"  📊 Archive: {len(df)} hours")
        
        return df
    
    def _fetch_forecast_weather(self) -> pd.DataFrame:
        """Fetch recent and forecast weather from Forecast API"""
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": "temperature_2m",
            "past_days": 7,  # Include recent past to fill gaps
            "forecast_days": 7,  # Future forecast
            "timezone": self.timezone
        }
        
        try:
            response = requests.get(forecast_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Weather Forecast API request failed: {e}")
            raise Exception(f"Weather Forecast API Error: {e}")
        
        if 'hourly' not in data:
            logger.warning("No forecast weather data available")
            return pd.DataFrame()
        
        df = pd.DataFrame({
            'timestamp': pd.to_datetime(data['hourly']['time']),
            'temperature': data['hourly']['temperature_2m']
        })
        
        df = df.set_index('timestamp')
        logger.info(f"  📊 Forecast: {len(df)} hours")
        
        return df
    
    def fetch_all_data(self, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Fetch both energy and weather data
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Tuple of (energy_df, weather_df)
        """
        energy_df = self.get_energy_data(start_date, end_date)
        
        # Extend weather fetch to include forecast for future predictions
        weather_end = (datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
        weather_df = self.get_weather_data(start_date, weather_end)
        
        return energy_df, weather_df
