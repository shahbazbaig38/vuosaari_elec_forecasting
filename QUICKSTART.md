# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### 1. Environment Setup (2 minutes)

```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model (2 minutes)

```powershell
# Train model with default settings
python src\train.py

# Or with custom date range
python src\train.py --start-date 2020-01-01 --train-until 2025-12-31
```

### 3. Start the Services (1 minute)

**Option A: FastAPI Backend**
```powershell
cd api
uvicorn main:app --reload --port 8000
```
Access API docs at: http://localhost:8000/docs

**Option B: Streamlit Dashboard**
```powershell
streamlit run ui\streamlit_app.py
```
Access dashboard at: http://localhost:8501

---

## 📊 Explore the Data

Open the Jupyter notebook for detailed analysis:
```powershell
jupyter notebook notebooks\01_eda_and_model_selection.ipynb
```

---

## 🧪 Run Tests

```powershell
# Run all tests
pytest tests\ -v

# Run with coverage
pytest tests\ --cov=src --cov-report=html
```

---

## 🔧 Common Commands

### Training
```powershell
# Basic training
python src\train.py

# With specific dates
python src\train.py --start-date 2023-01-01 --train-until 2025-12-31

# With custom test size
python src\train.py --test-size 1000
```

### API Testing
```powershell
# Test prediction endpoint
curl -X POST "http://localhost:8000/predict" -H "Content-Type: application/json" -d "{\"hours\": 24}"

# Get model info
curl http://localhost:8000/model/info

# Health check
curl http://localhost:8000/health
```

---

## 📁 Project Structure Overview

```
vuosaari_elec_forecasting/
├── src/                    # Core ML modules
│   ├── train.py           # Training script
│   ├── model.py           # Model class
│   ├── data_fetcher.py    # API data fetching
│   └── feature_engineering.py
├── api/                    # FastAPI backend
│   └── main.py
├── ui/                     # Streamlit dashboard
│   └── streamlit_app.py
├── notebooks/              # Jupyter notebooks
│   └── 01_eda_and_model_selection.ipynb
├── config/                 # Configuration
│   └── config.yaml
└── tests/                  # Unit tests
```

---

## 🎯 Next Steps

1. ✅ Train your first model
2. ✅ Explore the Jupyter notebook
3. ✅ Try the Streamlit dashboard
4. ✅ Test the API endpoints
5. ✅ Customize the configuration

---

## 💡 Tips

- **Model not found?** Run `python src\train.py` first
- **API errors?** Check if the model is trained
- **Port in use?** Change port: `--port 8001` or `--server.port 8502`
- **Need help?** Check README.md for detailed documentation

---

**Happy Forecasting! ⚡**
