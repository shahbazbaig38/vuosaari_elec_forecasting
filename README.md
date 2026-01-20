# ⚡ Vuosaari Electricity Consumption Forecasting System

A production-ready machine learning system for predicting electricity consumption at Vuosaari location (Helsinki) using historical consumption data and weather forecasts.

## 📋 Problem Definition

Electricity consumption forecasting is critical for:
- **Energy Grid Management**: Optimizing power distribution and preventing overloads
- **Cost Optimization**: Reducing operational costs through better demand prediction
- **Sustainability**: Enabling better integration of renewable energy sources
- **Resource Planning**: Helping facility managers plan maintenance and operations

This system predicts hourly electricity consumption for the next 24 hours at Vuosaari location (ID: 4438) in Helsinki, Finland.

## 💰 Financial Goals

1. **Cost Reduction**: Reduce energy procurement costs by 10-15% through accurate demand forecasting
2. **Peak Load Management**: Minimize peak demand charges by predicting and managing consumption spikes
3. **Operational Efficiency**: Reduce manual forecasting effort by 80% through automation

## 🎯 Key Features

- **Real-time Data Integration**: Fetches live electricity consumption from Nuuka API
- **Weather-Aware Predictions**: Incorporates temperature forecasts from Open-Meteo
- **24-Hour Forecasting**: Provides hourly predictions for the next day
- **Model Experimentation**: Jupyter notebook for EDA and model selection
- **REST API**: FastAPI backend for serving predictions
- **Interactive UI**: Streamlit dashboard for visualization and monitoring
- **Production-Ready**: Modular architecture, error handling, logging, and testing

## 📁 Project Structure

```
vuosaari_elec_forecasting/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── config/
│   └── config.yaml                   # Configuration parameters
├── data/
│   ├── raw/                          # Raw data from APIs
│   ├── processed/                    # Processed/cleaned data
│   └── predictions/                  # Model predictions
├── models/
│   └── trained_model.pkl             # Saved trained model
├── notebooks/
│   └── 01_eda_and_model_selection.ipynb  # Exploratory analysis & experiments
├── src/
│   ├── __init__.py
│   ├── config.py                     # Configuration management
│   ├── data_fetcher.py              # API data fetching
│   ├── feature_engineering.py       # Feature creation
│   ├── model.py                     # Model training & prediction
│   ├── utils.py                     # Utility functions
│   └── train.py                     # Training pipeline
├── api/
│   ├── __init__.py
│   ├── main.py                      # FastAPI application
│   └── schemas.py                   # Pydantic models
├── ui/
│   └── streamlit_app.py             # Streamlit dashboard
├── tests/
│   ├── __init__.py
│   ├── test_data_fetcher.py
│   ├── test_feature_engineering.py
│   └── test_model.py
└── logs/                             # Application logs
```

## 🔧 Technical Architecture

### Data Pipeline
1. **Data Collection**: Fetch electricity consumption (Nuuka API) and weather data (Open-Meteo API)
2. **Data Processing**: Clean, merge, and resample to hourly frequency
3. **Feature Engineering**: Create temporal and lag features
4. **Model Training**: Train Random Forest Regressor on historical data
5. **Prediction**: Generate 24-hour forecasts with weather integration

### Model Features
- `hour`: Hour of day (0-23)
- `dayofweek`: Day of week (0-6)
- `month_sin`: Cyclical month feature
- `temperature`: Temperature at 2m height (°C)
- `lag_24h`: Consumption 24 hours ago (kWh)
- `lag_48h`: Consumption 48 hours ago (kWh)
- `lag_168h`: Consumption 168 hours ago (kWh)

### Technology Stack
- **ML Framework**: scikit-learn (Random Forest Regressor)
- **API Framework**: FastAPI
- **UI Framework**: Streamlit
- **Data Processing**: pandas, numpy
- **HTTP Requests**: requests
- **Model Persistence**: joblib
- **Configuration**: PyYAML

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Internet connection (for API access)

### Environment Setup

1. **Clone the repository** (or navigate to project directory):
```bash
cd vuosaari_elec_forecasting
```

2. **Create virtual environment**:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Training the Model

Train the model on historical data:
```bash
python src/train.py
```

Optional arguments:
```bash
python src/train.py --start-date 2020-01-01 --train-until 2025-12-31
```

This will:
- Fetch historical data from APIs
- Engineer features
- Train the Random Forest model
- Save the model to `models/trained_model.pkl`
- Display evaluation metrics

### Running the FastAPI Backend

Start the API server:
```bash
cd api
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Running the Streamlit UI

Start the Streamlit dashboard:
```bash
streamlit run ui/streamlit_app.py
```

The UI will open in your browser at `http://localhost:8501`

## 📊 API Endpoints

### GET `/`
Health check endpoint

**Response:**
```json
{
  "message": "Vuosaari Electricity Forecasting API",
  "status": "running"
}
```

### POST `/predict`
Generate 24-hour electricity consumption forecast

**Request Body:**
```json
{
  "hours": 24
}
```

**Response:**
```json
{
  "location_id": "4438",
  "location_name": "Vuosaari",
  "forecast_generated_at": "2026-01-20T10:26:30",
  "predictions": [
    {
      "timestamp": "2026-01-20T11:00:00",
      "predicted_consumption_kwh": 245.3,
      "temperature_celsius": -5.2
    },
    ...
  ],
  "model_info": {
    "model_type": "RandomForestRegressor",
    "features_used": ["hour", "dayofweek", "month", "temperature", "lag_24h"],
    "training_period": "2020-01-01 to 2025-12-31"
  }
}
```

### GET `/model/info`
Get information about the trained model

### GET `/historical`
Retrieve historical consumption data

## 🎨 Streamlit Dashboard Features

- **Real-time Predictions**: View 24-hour consumption forecasts
- **Historical Trends**: Analyze past consumption patterns
- **Weather Integration**: See temperature impact on consumption
- **Model Performance**: View evaluation metrics and feature importance
- **Data Download**: Export predictions to CSV
- **Interactive Charts**: Zoom, pan, and explore data visually

## 📈 Model Performance

Based on validation on recent data (last 500 hours):
- **Mean Absolute Error (MAE)**: ~15-25 kWh
- **R² Score**: ~0.85-0.92
- **Training Time**: ~30-60 seconds
- **Prediction Time**: <1 second for 24 hours

*Note: Performance metrics are updated after each training run*

## 🔬 Experimentation & Model Selection

The `notebooks/01_eda_and_model_selection.ipynb` notebook contains:
- Exploratory Data Analysis (EDA)
- Data quality assessment
- Feature correlation analysis
- Multiple model comparisons
- Hyperparameter tuning
- Model selection rationale

**Selected Model**: Random Forest Regressor
- **Reason**: Best balance of accuracy, speed, and interpretability
- **Advantages**: Handles non-linear patterns, robust to outliers, feature importance

## 🧪 Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📝 Configuration

Edit `config/config.yaml` to customize:
- API endpoints and credentials
- Location coordinates
- Model hyperparameters
- Logging settings
- Data storage paths

## 🔒 Environment Variables

Create a `.env` file for sensitive data (optional):
```


## 🐛 Troubleshooting

### API Connection Issues
- Check internet connection
- Verify API endpoints are accessible
- Check if APIs have rate limits

### Model Training Fails
- Ensure sufficient historical data is available
- Check date ranges are valid
- Verify data quality (no excessive missing values)

### Port Already in Use
```bash
# Change port for FastAPI
uvicorn main:app --port 8001

# Change port for Streamlit
streamlit run ui/streamlit_app.py --server.port 8502
```

## 📚 References

- **Nuuka API**: Helsinki Open API for energy data
- **Open-Meteo**: Free weather API for historical and forecast data
- **Location**: Vuosaari, Helsinki (ID: 4438, Lat: 60.1699, Lon: 24.9384)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👥 Authors

- **Shahbaz Baig** 

## 🙏 Acknowledgments

- Helsinki Region Environmental Services for providing open energy data
- Open-Meteo for free weather data access

---

**Last Updated**: 2026-01-20
**Version**: 1.0.0
