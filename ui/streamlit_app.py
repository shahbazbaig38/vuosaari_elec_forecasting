"""
Streamlit Dashboard for Electricity Consumption Forecasting

Interactive UI for visualizing predictions and historical data
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.model import ElectricityForecastModel

# Page configuration
st.set_page_config(
    page_title="⚡ Vuosaari Electricity Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize configuration
@st.cache_resource
def load_config():
    """Load configuration"""
    return get_config()

config = load_config()

# Initialize model
@st.cache_resource
def load_model():
    """Load trained model"""
    try:
        model = ElectricityForecastModel()
        model.load(config.model_save_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Header
st.markdown('<div class="main-header">⚡ Vuosaari Electricity Forecasting Dashboard</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/300x100/1f77b4/ffffff?text=Vuosaari+Energy", width=300)
    
    st.markdown("### 🎛️ Control Panel")
    
    page = st.radio(
        "Select View",
        ["📊 Dashboard", "🔮 Forecast", "📈 Historical Analysis", "🤖 Model Info"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    st.markdown("### ℹ️ About")
    st.markdown("""
    This dashboard provides real-time electricity consumption forecasting 
    for Vuosaari location using machine learning.
    
    **Features:**
    - 24-hour consumption forecasts
    - Historical data analysis
    - Weather integration
    - Model performance metrics
    """)
    
    st.markdown("---")
    st.markdown("**Location:** Vuosaari, Helsinki")
    st.markdown(f"**Location ID:** {config.location_id}")
    st.markdown(f"**Coordinates:** {config.latitude}°N, {config.longitude}°E")

# Main content area
if page == "📊 Dashboard":
    st.header("📊 Overview Dashboard")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ No trained model available. Please train a model first.")
        st.code("python src/train.py", language="bash")
        st.stop()
    
    # Generate quick forecast
    with st.spinner("Generating forecast..."):
        try:
            # Fetch data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            fetcher = DataFetcher()
            energy_df, weather_df = fetcher.fetch_all_data(start_date, end_date)
            
            # Prepare prediction
            future_start = pd.Timestamp.now().ceil('h')
            future_dates = pd.date_range(start=future_start, periods=24, freq='h')
            
            # Use config for feature engineering to match trained model
            engineer = FeatureEngineer(
                lag_hours=config.get('features.lag_hours', [24]),
                rolling_windows=config.get('features.rolling_windows', [])
            )
            df_future = engineer.prepare_prediction_data(
                future_dates=future_dates,
                weather_df=weather_df,
                historical_consumption=energy_df
            )
            
            # Predict
            features = model.feature_names
            predictions = model.predict(df_future[features])
            df_future['predicted_consumption'] = predictions
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_consumption = predictions.mean()
                st.metric(
                    "Avg. Forecast (24h)",
                    f"{avg_consumption:.1f} kWh",
                    delta=None
                )
            
            with col2:
                peak_consumption = predictions.max()
                peak_hour = future_dates[predictions.argmax()].strftime('%H:%M')
                st.metric(
                    "Peak Consumption",
                    f"{peak_consumption:.1f} kWh",
                    delta=f"at {peak_hour}"
                )
            
            with col3:
                current_temp = df_future['temperature'].iloc[0] if 'temperature' in df_future.columns else 0
                st.metric(
                    "Current Temperature",
                    f"{current_temp:.1f}°C",
                    delta=None
                )
            
            with col4:
                model_info = model.get_model_info()
                val_mae = model_info.get('val_metrics', {}).get('validation_mae', 0)
                st.metric(
                    "Model MAE",
                    f"{val_mae:.1f} kWh",
                    delta=None
                )
            
            st.markdown("---")
            
            # Forecast chart
            st.subheader("📈 24-Hour Consumption Forecast")
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=predictions,
                mode='lines+markers',
                name='Predicted Consumption',
                line=dict(color='#1f77b4', width=3),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title="Electricity Consumption Forecast",
                xaxis_title="Time",
                yaxis_title="Consumption (kWh)",
                hovermode='x unified',
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, width='stretch')
            
            # Recent historical data
            st.subheader("📊 Recent Historical Data (Last 7 Days)")
            
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=energy_df.index,
                y=energy_df['consumption'],
                mode='lines',
                name='Actual Consumption',
                line=dict(color='#2ca02c', width=2)
            ))
            
            fig2.update_layout(
                title="Historical Consumption",
                xaxis_title="Date",
                yaxis_title="Consumption (kWh)",
                hovermode='x unified',
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig2, width='stretch')
            
        except Exception as e:
            st.error(f"Error generating dashboard: {e}")
            import traceback
            st.code(traceback.format_exc())

elif page == "🔮 Forecast":
    st.header("🔮 Generate Custom Forecast")
    
    model = load_model()
    
    if model is None:
        st.error("⚠️ No trained model available.")
        st.stop()
    
    # Forecast parameters
    col1, col2 = st.columns([1, 3])
    
    with col1:
        forecast_hours = st.slider("Forecast Hours", min_value=1, max_value=168, value=24, step=1)
        
        if st.button("🔮 Generate Forecast", type="primary"):
            with st.spinner(f"Generating {forecast_hours}-hour forecast..."):
                try:
                    # Fetch data
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                    
                    fetcher = DataFetcher()
                    energy_df, weather_df = fetcher.fetch_all_data(start_date, end_date)
                    
                    # Prepare prediction
                    future_start = pd.Timestamp.now().ceil('h')
                    future_dates = pd.date_range(start=future_start, periods=forecast_hours, freq='h')
                    
                    # Use config for feature engineering to match trained model
                    engineer = FeatureEngineer(
                        lag_hours=config.get('features.lag_hours', [24]),
                        rolling_windows=config.get('features.rolling_windows', [])
                    )
                    df_future = engineer.prepare_prediction_data(
                        future_dates=future_dates,
                        weather_df=weather_df,
                        historical_consumption=energy_df
                    )
                    
                    # Predict
                    features = model.feature_names
                    predictions = model.predict(df_future[features])
                    df_future['predicted_consumption'] = predictions
                    
                    # Store in session state
                    st.session_state['forecast_df'] = df_future
                    st.session_state['forecast_dates'] = future_dates
                    st.session_state['predictions'] = predictions
                    
                    st.success(f"✅ Forecast generated for {forecast_hours} hours!")
                    
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Display forecast if available
    if 'predictions' in st.session_state:
        df_future = st.session_state['forecast_df']
        future_dates = st.session_state['forecast_dates']
        predictions = st.session_state['predictions']
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=predictions,
            mode='lines+markers',
            name='Predicted Consumption',
            line=dict(color='#ff7f0e', width=3),
            marker=dict(size=5),
            fill='tozeroy',
            fillcolor='rgba(255, 127, 14, 0.1)'
        ))
        
        if 'temperature' in df_future.columns:
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=df_future['temperature'],
                mode='lines',
                name='Temperature',
                line=dict(color='#d62728', width=2, dash='dash'),
                yaxis='y2'
            ))
        
        fig.update_layout(
            title=f"{len(predictions)}-Hour Electricity Consumption Forecast",
            xaxis_title="Time",
            yaxis_title="Consumption (kWh)",
            yaxis2=dict(
                title="Temperature (°C)",
                overlaying='y',
                side='right'
            ),
            hovermode='x unified',
            height=500,
            template="plotly_white",
            legend=dict(x=0, y=1.1, orientation='h')
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Statistics
        st.subheader("📊 Forecast Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average", f"{predictions.mean():.1f} kWh")
        col2.metric("Maximum", f"{predictions.max():.1f} kWh")
        col3.metric("Minimum", f"{predictions.min():.1f} kWh")
        col4.metric("Std Dev", f"{predictions.std():.1f} kWh")
        
        # Data table
        st.subheader("📋 Forecast Data")
        
        display_df = pd.DataFrame({
            'Timestamp': future_dates,
            'Predicted Consumption (kWh)': predictions.round(2),
            'Temperature (°C)': df_future['temperature'].round(1) if 'temperature' in df_future.columns else None
        })
        
        st.dataframe(display_df, width='stretch', height=400)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

elif page == "📈 Historical Analysis":
    st.header("📈 Historical Data Analysis")
    
    # Date range selector
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30),
            max_value=datetime.now()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            max_value=datetime.now()
        )
    
    if st.button("📊 Load Historical Data", type="primary"):
        with st.spinner("Loading historical data..."):
            try:
                fetcher = DataFetcher()
                energy_df, weather_df = fetcher.fetch_all_data(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                # Merge data
                df = energy_df.join(weather_df, how='inner')
                
                st.session_state['historical_df'] = df
                st.success(f"✅ Loaded {len(df)} data points")
                
            except Exception as e:
                st.error(f"Error loading data: {e}")
    
    # Display historical data if available
    if 'historical_df' in st.session_state:
        df = st.session_state['historical_df']
        
        # Summary statistics
        st.subheader("📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Avg Consumption", f"{df['consumption'].mean():.1f} kWh")
        col2.metric("Max Consumption", f"{df['consumption'].max():.1f} kWh")
        col3.metric("Min Consumption", f"{df['consumption'].min():.1f} kWh")
        col4.metric("Total Hours", f"{len(df)}")
        
        # Consumption over time
        st.subheader("📈 Consumption Over Time")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['consumption'],
            mode='lines',
            name='Consumption',
            line=dict(color='#2ca02c', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Consumption (kWh)",
            hovermode='x unified',
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Temperature vs Consumption
        if 'temperature' in df.columns:
            st.subheader("🌡️ Temperature vs Consumption")
            
            fig2 = px.scatter(
                df,
                x='temperature',
                y='consumption',
                color='consumption',
                color_continuous_scale='Viridis',
                labels={'temperature': 'Temperature (°C)', 'consumption': 'Consumption (kWh)'},
                title="Consumption vs Temperature Relationship"
            )
            
            fig2.update_layout(height=400, template="plotly_white")
            st.plotly_chart(fig2, width='stretch')
        
        # Hourly patterns
        st.subheader("⏰ Hourly Consumption Patterns")
        
        df['hour'] = df.index.hour
        hourly_avg = df.groupby('hour')['consumption'].mean()
        
        fig3 = go.Figure()
        
        fig3.add_trace(go.Bar(
            x=hourly_avg.index,
            y=hourly_avg.values,
            marker_color='#9467bd'
        ))
        
        fig3.update_layout(
            title="Average Consumption by Hour of Day",
            xaxis_title="Hour",
            yaxis_title="Average Consumption (kWh)",
            height=400,
            template="plotly_white"
        )
        
        st.plotly_chart(fig3, width='stretch')

elif page == "🤖 Model Info":
    st.header("🤖 Model Information")
    
    model = load_model()
    
    if model is None:
        st.error("⚠️ No trained model available.")
        st.stop()
    
    model_info = model.get_model_info()
    
    # Model details
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Model Details")
        st.markdown(f"**Model Type:** {model_info.get('model_type', 'Unknown')}")
        st.markdown(f"**Number of Features:** {model_info.get('n_features', 'Unknown')}")
        st.markdown(f"**Training Samples:** {model_info.get('n_train_samples', 'Unknown')}")
        st.markdown(f"**Training Time:** {model_info.get('training_time_seconds', 0):.2f} seconds")
        st.markdown(f"**Trained At:** {model_info.get('trained_at', 'Unknown')}")
    
    with col2:
        st.subheader("📊 Performance Metrics")
        
        val_metrics = model_info.get('val_metrics', {})
        
        if val_metrics:
            st.metric("Validation MAE", f"{val_metrics.get('validation_mae', 0):.2f} kWh")
            st.metric("Validation RMSE", f"{val_metrics.get('validation_rmse', 0):.2f} kWh")
            st.metric("Validation R²", f"{val_metrics.get('validation_r2', 0):.4f}")
            st.metric("Validation MAPE", f"{val_metrics.get('validation_mape', 0):.2f}%")
        else:
            st.info("No validation metrics available")
    
    # Features used
    st.subheader("🔧 Features Used")
    
    features = model_info.get('feature_names', [])
    if features:
        # Display in columns
        cols = st.columns(4)
        for i, feature in enumerate(features):
            cols[i % 4].markdown(f"✓ {feature}")
    
    # Feature importance
    st.subheader("📊 Feature Importance")
    
    importance_df = model.get_feature_importance()
    
    if not importance_df.empty:
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=importance_df['importance'].head(15),
            y=importance_df['feature'].head(15),
            orientation='h',
            marker_color='#17becf'
        ))
        
        fig.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=500,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, width='stretch')
        
        # Full table
        with st.expander("📋 View All Feature Importances"):
            st.dataframe(importance_df, width='stretch')
    else:
        st.info("Feature importance not available for this model type")
    
    # Model parameters
    st.subheader("⚙️ Model Hyperparameters")
    
    params = model_info.get('model_params', {})
    if params:
        st.json(params)
    else:
        st.info("No hyperparameters available")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>⚡ Vuosaari Electricity Forecasting System v1.0.0</p>
    <p>Powered by Machine Learning | Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)
