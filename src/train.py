"""
Training Pipeline Script

Command-line script to train the electricity forecasting model
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging, print_section_header
from src.data_fetcher import DataFetcher
from src.feature_engineering import FeatureEngineer
from src.model import train_and_save_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train electricity consumption forecasting model"
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for training data (YYYY-MM-DD). Default: from config'
    )
    
    parser.add_argument(
        '--train-until',
        type=str,
        default=None,
        help='End date for training (YYYY-MM-DD). Default: current date'
    )
    
    parser.add_argument(
        '--test-size',
        type=int,
        default=500,
        help='Number of hours to use for validation. Default: 500'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config file. Default: config/config.yaml'
    )
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = get_config(args.config)
    
    # Setup logging
    setup_logging(config.log_level, config.log_file)
    
    print_section_header("⚡ VUOSAARI ELECTRICITY FORECASTING - TRAINING")
    
    # Determine date range
    start_date = args.start_date or config.start_date
    train_end_date = args.train_until
    
    # If no train_until specified, fetch until today
    if train_end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    else:
        end_date = train_end_date
    
    print(f"📅 Training Period:")
    print(f"   Start Date: {start_date}")
    print(f"   End Date:   {end_date}")
    print(f"   Test Size:  {args.test_size} hours\n")
    
    try:
        # Step 1: Fetch Data
        print_section_header("STEP 1: DATA FETCHING")
        fetcher = DataFetcher()
        energy_df, weather_df = fetcher.fetch_all_data(start_date, end_date)
        
        # Step 2: Feature Engineering
        print_section_header("STEP 2: FEATURE ENGINEERING")
        engineer = FeatureEngineer(
            lag_hours=config.get('features.lag_hours', [24]),
            rolling_windows=config.get('features.rolling_windows', [])
        )
        
        df = engineer.prepare_training_data(energy_df, weather_df)
        
        print(f"\n📊 Dataset Summary:")
        print(f"   Total Samples: {len(df)}")
        print(f"   Features: {len(df.columns) - 1}")  # -1 for target
        print(f"   Date Range: {df.index.min()} to {df.index.max()}")
        print(f"\n   Feature Columns: {', '.join(df.columns.tolist())}\n")
        
        # Step 3: Train Model
        print_section_header("STEP 3: MODEL TRAINING")
        
        features = config.features
        target = config.target
        
        # Verify all features exist in dataframe
        missing_features = set(features) - set(df.columns)
        if missing_features:
            print(f"⚠️  Warning: Features not in dataset: {missing_features}")
            print(f"   Using available features from config that exist in data")
            features = [f for f in features if f in df.columns]
        
        print(f"Using features: {features}\n")
        
        model = train_and_save_model(
            df=df,
            features=features,
            target=target,
            train_end_date=train_end_date,
            test_size=args.test_size,
            save_path=config.model_save_path
        )
        
        # Step 4: Summary
        print_section_header("TRAINING SUMMARY")
        model_info = model.get_model_info()
        
        print(f"✅ Model Type: {model_info.get('model_type')}")
        print(f"✅ Features Used: {model_info.get('n_features')}")
        print(f"✅ Training Samples: {model_info.get('n_train_samples')}")
        print(f"✅ Training Time: {model_info.get('training_time_seconds', 0):.2f} seconds")
        print(f"✅ Model Saved: {config.model_save_path}")
        
        train_metrics = model_info.get('train_metrics', {})
        val_metrics = model_info.get('val_metrics', {})
        
        if train_metrics:
            print(f"\n📈 Training Performance:")
            print(f"   MAE:  {train_metrics.get('training_mae', 0):.2f} kWh")
            print(f"   RMSE: {train_metrics.get('training_rmse', 0):.2f} kWh")
            print(f"   R²:   {train_metrics.get('training_r2', 0):.4f}")
        
        if val_metrics:
            print(f"\n📊 Validation Performance:")
            print(f"   MAE:  {val_metrics.get('validation_mae', 0):.2f} kWh")
            print(f"   RMSE: {val_metrics.get('validation_rmse', 0):.2f} kWh")
            print(f"   R²:   {val_metrics.get('validation_r2', 0):.4f}")
        
        print("\n" + "=" * 60)
        print("  🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60 + "\n")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
