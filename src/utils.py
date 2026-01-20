"""
Utility Functions Module

Common utility functions used across the project
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import pandas as pd


def setup_logging(log_level: str = "INFO", log_file: Optional[Path] = None):
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file. If None, logs only to console.
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Create handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def validate_date_format(date_string: str) -> bool:
    """
    Validate if string is in YYYY-MM-DD format
    
    Args:
        date_string: Date string to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        datetime.strptime(date_string, '%Y-%m-%d')
        return True
    except ValueError:
        return False


def get_date_range(start_date: str, end_date: str) -> int:
    """
    Calculate number of days between two dates
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        Number of days
    """
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')
    return (end - start).days


def save_dataframe(df: pd.DataFrame, filepath: Path, format: str = 'csv'):
    """
    Save DataFrame to file
    
    Args:
        df: DataFrame to save
        filepath: Path to save file
        format: File format ('csv', 'parquet', 'json')
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'csv':
        df.to_csv(filepath)
    elif format == 'parquet':
        df.to_parquet(filepath)
    elif format == 'json':
        df.to_json(filepath, orient='records', date_format='iso')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Saved DataFrame to {filepath}")


def load_dataframe(filepath: Path, format: str = 'csv') -> pd.DataFrame:
    """
    Load DataFrame from file
    
    Args:
        filepath: Path to file
        format: File format ('csv', 'parquet', 'json')
        
    Returns:
        Loaded DataFrame
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    if format == 'csv':
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    elif format == 'parquet':
        df = pd.read_parquet(filepath)
    elif format == 'json':
        df = pd.read_json(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logging.info(f"Loaded DataFrame from {filepath}")
    return df


def format_consumption(value: float) -> str:
    """
    Format consumption value for display
    
    Args:
        value: Consumption in kWh
        
    Returns:
        Formatted string
    """
    return f"{value:.2f} kWh"


def format_temperature(value: float) -> str:
    """
    Format temperature value for display
    
    Args:
        value: Temperature in Celsius
        
    Returns:
        Formatted string
    """
    return f"{value:.1f}°C"


def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    
    Args:
        old_value: Original value
        new_value: New value
        
    Returns:
        Percentage change
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100


def get_timestamp_string() -> str:
    """
    Get current timestamp as formatted string
    
    Returns:
        Timestamp string (YYYYMMDD_HHMMSS)
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def print_section_header(title: str, char: str = "="):
    """
    Print a formatted section header
    
    Args:
        title: Section title
        char: Character to use for border
    """
    border = char * 60
    print(f"\n{border}")
    print(f"  {title}")
    print(f"{border}\n")
