"""
Pydantic Schemas for API Request/Response Models
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for prediction endpoint"""
    hours: int = Field(default=24, ge=1, le=168, description="Number of hours to forecast (1-168)")


class PredictionPoint(BaseModel):
    """Single prediction point"""
    timestamp: datetime
    predicted_consumption_kwh: float
    temperature_celsius: Optional[float] = None


class ModelInfo(BaseModel):
    """Model metadata"""
    model_type: str
    features_used: List[str]
    training_period: str
    trained_at: Optional[str] = None
    n_features: Optional[int] = None
    validation_mae: Optional[float] = None
    validation_r2: Optional[float] = None


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint"""
    location_id: str
    location_name: str
    forecast_generated_at: datetime
    predictions: List[PredictionPoint]
    model_info: ModelInfo


class HealthResponse(BaseModel):
    """Health check response"""
    message: str
    status: str
    version: str
    timestamp: datetime


class HistoricalDataRequest(BaseModel):
    """Request for historical data"""
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str = Field(..., description="End date (YYYY-MM-DD)")


class HistoricalDataPoint(BaseModel):
    """Single historical data point"""
    timestamp: datetime
    consumption_kwh: float
    temperature_celsius: Optional[float] = None


class HistoricalDataResponse(BaseModel):
    """Response with historical data"""
    location_id: str
    location_name: str
    start_date: str
    end_date: str
    data_points: List[HistoricalDataPoint]
    total_points: int


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime
