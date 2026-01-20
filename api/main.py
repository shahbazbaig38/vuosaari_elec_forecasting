"""
FastAPI Application for Electricity Consumption Forecasting

Provides REST API endpoints for model predictions and data access
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import get_config
from src.utils import setup_logging
from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.model import ElectricityForecastModel

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    PredictionPoint,
    ModelInfo,
    HealthResponse,
    HistoricalDataRequest,
    HistoricalDataResponse,
    HistoricalDataPoint,
    ErrorResponse
)

# Initialize FastAPI app
app = FastAPI(
    title="Vuosaari Electricity Forecasting API",
    description="REST API for electricity consumption forecasting using ML",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
config = None
model = None
logger = None


@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    global config, model, logger
    
    # Load configuration
    config = get_config()
    
    # Setup logging
    setup_logging(config.log_level, config.log_file)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("  Starting Vuosaari Electricity Forecasting API")
    logger.info("=" * 60)
    
    # Load trained model
    try:
        model = ElectricityForecastModel()
        model.load(config.model_save_path)
        logger.info("✅ Model loaded successfully")
    except FileNotFoundError:
        logger.warning("⚠️  No trained model found. Please train a model first.")
        logger.warning(f"   Expected location: {config.model_save_path}")
        model = None
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        model = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down API server...")


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        message="Vuosaari Electricity Forecasting API",
        status="running",
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    model_status = "loaded" if model is not None else "not_loaded"
    
    return HealthResponse(
        message=f"API is running. Model status: {model_status}",
        status="healthy" if model is not None else "degraded",
        version="1.0.0",
        timestamp=datetime.now()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Generate electricity consumption forecast
    
    Args:
        request: Prediction request with number of hours
        
    Returns:
        Forecast predictions with metadata
    """
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please train a model first."
        )
    
    try:
        logger.info(f"🔮 Generating {request.hours}-hour forecast...")
        
        # Fetch recent historical data for lag features
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        fetcher = DataFetcher()
        energy_df, weather_df = fetcher.fetch_all_data(start_date, end_date)
        
        # Create future timestamps
        future_start = pd.Timestamp.now().ceil('h')
        future_dates = pd.date_range(start=future_start, periods=request.hours, freq='h')
        
        # Prepare prediction features
        engineer = FeatureEngineer(
            lag_hours=config.get('features.lag_hours', [24]),
            rolling_windows=config.get('features.rolling_windows', [])
        )
        
        df_future = engineer.prepare_prediction_data(
            future_dates=future_dates,
            weather_df=weather_df,
            historical_consumption=energy_df
        )
        
        # Make predictions
        features = model.feature_names
        predictions = model.predict(df_future[features])
        
        # Prepare response
        prediction_points = []
        for i, (timestamp, pred) in enumerate(zip(future_dates, predictions)):
            temp = df_future.loc[timestamp, 'temperature'] if 'temperature' in df_future.columns else None
            
            prediction_points.append(
                PredictionPoint(
                    timestamp=timestamp,
                    predicted_consumption_kwh=float(pred),
                    temperature_celsius=float(temp) if temp is not None else None
                )
            )
        
        # Get model info
        model_info_dict = model.get_model_info()
        training_start = config.start_date
        training_end = model_info_dict.get('trained_at', 'Unknown')
        
        val_metrics = model_info_dict.get('val_metrics', {})
        
        model_info = ModelInfo(
            model_type=model_info_dict.get('model_type', 'Unknown'),
            features_used=model_info_dict.get('feature_names', []),
            training_period=f"{training_start} to {training_end}",
            trained_at=model_info_dict.get('trained_at'),
            n_features=model_info_dict.get('n_features'),
            validation_mae=val_metrics.get('validation_mae'),
            validation_r2=val_metrics.get('validation_r2')
        )
        
        response = PredictionResponse(
            location_id=config.location_id,
            location_name="Vuosaari",
            forecast_generated_at=datetime.now(),
            predictions=prediction_points,
            model_info=model_info
        )
        
        logger.info(f"✅ Forecast generated successfully")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the trained model"""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    model_info_dict = model.get_model_info()
    training_start = config.start_date
    training_end = model_info_dict.get('trained_at', 'Unknown')
    
    val_metrics = model_info_dict.get('val_metrics', {})
    
    return ModelInfo(
        model_type=model_info_dict.get('model_type', 'Unknown'),
        features_used=model_info_dict.get('feature_names', []),
        training_period=f"{training_start} to {training_end}",
        trained_at=model_info_dict.get('trained_at'),
        n_features=model_info_dict.get('n_features'),
        validation_mae=val_metrics.get('validation_mae'),
        validation_r2=val_metrics.get('validation_r2')
    )


@app.post("/historical", response_model=HistoricalDataResponse)
async def get_historical_data(request: HistoricalDataRequest):
    """
    Retrieve historical electricity consumption data
    
    Args:
        request: Date range for historical data
        
    Returns:
        Historical consumption and weather data
    """
    try:
        logger.info(f"📊 Fetching historical data: {request.start_date} to {request.end_date}")
        
        fetcher = DataFetcher()
        energy_df, weather_df = fetcher.fetch_all_data(request.start_date, request.end_date)
        
        # Merge data
        df = energy_df.join(weather_df, how='inner')
        
        # Create response
        data_points = []
        for timestamp, row in df.iterrows():
            data_points.append(
                HistoricalDataPoint(
                    timestamp=timestamp,
                    consumption_kwh=float(row['consumption']),
                    temperature_celsius=float(row['temperature']) if 'temperature' in row else None
                )
            )
        
        response = HistoricalDataResponse(
            location_id=config.location_id,
            location_name="Vuosaari",
            start_date=request.start_date,
            end_date=request.end_date,
            data_points=data_points,
            total_points=len(data_points)
        )
        
        logger.info(f"✅ Retrieved {len(data_points)} historical data points")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Historical data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve historical data: {str(e)}"
        )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal Server Error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


if __name__ == "__main__":
    import uvicorn
    
    # Run server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
