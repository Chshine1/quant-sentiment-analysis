#!/usr/bin/env python3
"""
Main Execution Script for Stock Return Prediction

This script orchestrates the full pipeline:
1. Data preprocessing and sentiment computation
2. Feature engineering
3. Model training for each horizon (k=1, 2, 3, 7)
4. Evaluation and results reporting

Usage:
    python main.py --input data/tweets.csv --output output/
    
    # With preprocessed data
    python main.py --input data/daily_aggregated.csv --skip-preprocessing
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

import torch

sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing_and_sentiment import DataPreprocessor
from src.feature_engineering import FeatureEngineer, DatasetBuilder, create_lookback_data
from src.training_pipeline import StockReturnTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration for the pipeline."""
    
    INPUT_CSV = 'data/tweets.csv'
    OUTPUT_DIR = 'output'
    CACHE_DIR = 'data/cache'
    
    TRAIN_START = '2018-07-01'
    TRAIN_END = '2018-09-15'
    TEST_START = '2018-10-01'
    TEST_END = '2018-10-31'
    VAL_RATIO = 0.2
    
    T = 7
    LAMBDA = 0.3
    
    LSTM_HIDDEN_SIZE = 64
    LSTM_NUM_LAYERS = 1
    HIDDEN_DIMS = (128, 64, 32)
    DROPOUT = 0.3
    
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 32
    N_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 5
    
    HORIZONS = ['1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN']
    
    COMPUTE_TEXTBLOB = True
    COMPUTE_FINBERT = True
    FORCE_RECOMPUTE = False


def setup_directories(output_dir: str, cache_dir: str):
    """Create necessary directories."""
    for d in [output_dir, cache_dir, 'logs']:
        Path(d).mkdir(parents=True, exist_ok=True)


def preprocess_data(config: Config) -> pd.DataFrame:
    """
    Step 1: Preprocess data and compute sentiments.
    
    Args:
        config: Configuration object
    
    Returns:
        Processed daily aggregated DataFrame
    """
    logger.info("="*60)
    logger.info("STEP 1: DATA PREPROCESSING")
    logger.info("="*60)
    
    preprocessor = DataPreprocessor(cache_dir=config.CACHE_DIR)
    
    daily_df = preprocessor.process_full_pipeline(
        csv_path=config.INPUT_CSV,
        compute_textblob=config.COMPUTE_TEXTBLOB,
        compute_finbert=config.COMPUTE_FINBERT,
        force_recompute=config.FORCE_RECOMPUTE
    )
    
    logger.info(f"Preprocessed data shape: {daily_df.shape}")
    logger.info(f"Date range: {daily_df['DATE'].min()} to {daily_df['DATE'].max()}")
    logger.info(f"Unique stocks: {daily_df['STOCK_CODE'].nunique()}")
    
    return daily_df


def engineer_features(daily_df: pd.DataFrame, config: Config) -> tuple:
    """
    Step 2: Engineer features.
    
    Args:
        daily_df: Processed daily data
        config: Configuration object
    
    Returns:
        Tuple of (train_df, val_df, test_df, feature_engineer)
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 2: FEATURE ENGINEERING")
    logger.info("="*60)
    
    feature_engineer = FeatureEngineer()
    daily_df = feature_engineer.build_all_features(daily_df)
    
    daily_df, sequences = create_lookback_data(daily_df, feature_engineer, T=config.T)
    
    train_df, val_df, test_df = feature_engineer.create_temporal_split(
        daily_df,
        train_start=config.TRAIN_START,
        train_end=config.TRAIN_END,
        test_start=config.TEST_START,
        test_end=config.TEST_END,
        val_ratio=config.VAL_RATIO
    )
    
    logger.info(f"Training samples: {len(train_df)}")
    logger.info(f"Validation samples: {len(val_df)}")
    logger.info(f"Test samples: {len(test_df)}")
    
    return train_df, val_df, test_df, feature_engineer, sequences


def prepare_datasets(train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    feature_engineer: FeatureEngineer,
                    sequences: np.ndarray,
                    config: Config) -> tuple:
    """
    Step 3: Prepare datasets for training.
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 3: DATASET PREPARATION")
    logger.info("="*60)
    
    feature_engineer.fit_scalers(train_df)
    
    dataset_builder = DatasetBuilder(feature_engineer)
    
    logger.info("Preparing training dataset...")
    train_features = dataset_builder.build_dataset(train_df, '1_DAY_RETURN', sequences)
    train_indices = train_df.index.tolist()
    
    for key in train_features:
        if isinstance(train_features[key], np.ndarray) and len(train_features[key]) > 0:
            if key == 'sequences':
                continue
            logger.info(f"  {key}: shape {train_features[key].shape}")
    
    logger.info("Preparing validation dataset...")
    val_features = dataset_builder.build_dataset(val_df, '1_DAY_RETURN', sequences)
    
    logger.info("Preparing test dataset...")
    test_features = dataset_builder.build_dataset(test_df, '1_DAY_RETURN', sequences)
    
    return train_features, val_features, test_features


def train_models(train_features: dict,
                val_features: dict,
                test_features: dict,
                config: Config) -> pd.DataFrame:
    """
    Step 4: Train models for all horizons.
    """
    logger.info("\n" + "="*60)
    logger.info("STEP 4: MODEL TRAINING")
    logger.info("="*60)
    
    logger.info(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    trainer = StockReturnTrainer(
        output_dir=config.OUTPUT_DIR,
        lstm_hidden_size=config.LSTM_HIDDEN_SIZE,
        lstm_num_layers=config.LSTM_NUM_LAYERS,
        hidden_dims=config.HIDDEN_DIMS,
        dropout=config.DROPOUT,
        learning_rate=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        batch_size=config.BATCH_SIZE,
        n_epochs=config.N_EPOCHS,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE
    )
    
    results_df = trainer.train_all_horizons(
        train_data=train_features,
        val_data=val_features,
        test_data=test_features,
        horizons=config.HORIZONS
    )
    
    return results_df


def main():
    """Run the full pipeline."""
    parser = argparse.ArgumentParser(
        description='Stock Return Prediction with Twitter Sentiment Analysis'
    )
    
    parser.add_argument('--input', '-i', default=Config.INPUT_CSV,
                       help='Input CSV file path')
    parser.add_argument('--output', '-o', default=Config.OUTPUT_DIR,
                       help='Output directory')
    parser.add_argument('--cache-dir', '-c', default=Config.CACHE_DIR,
                       help='Cache directory for intermediate results')
    
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing step (use cached data)')
    parser.add_argument('--skip-textblob', action='store_true',
                       help='Skip TextBlob computation')
    parser.add_argument('--skip-finbert', action='store_true',
                       help='Skip FinBERT computation')
    parser.add_argument('--force-recompute', action='store_true',
                       help='Force recompute even if cached')
    
    parser.add_argument('--train-start', default=Config.TRAIN_START)
    parser.add_argument('--train-end', default=Config.TRAIN_END)
    parser.add_argument('--test-start', default=Config.TEST_START)
    parser.add_argument('--test-end', default=Config.TEST_END)
    
    parser.add_argument('--epochs', type=int, default=Config.N_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=Config.LEARNING_RATE)
    
    args = parser.parse_args()
    
    config = Config()
    config.INPUT_CSV = args.input
    config.OUTPUT_DIR = args.output
    config.CACHE_DIR = args.cache_dir
    config.TRAIN_START = args.train_start
    config.TRAIN_END = args.train_end
    config.TEST_START = args.test_start
    config.TEST_END = args.test_end
    config.N_EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.lr
    config.COMPUTE_TEXTBLOB = not args.skip_textblob
    config.COMPUTE_FINBERT = not args.skip_finbert
    config.FORCE_RECOMPUTE = args.force_recompute
    
    setup_directories(config.OUTPUT_DIR, config.CACHE_DIR)
    
    logger.info("="*60)
    logger.info("STOCK RETURN PREDICTION PIPELINE")
    logger.info(f"Start time: {datetime.now()}")
    logger.info("="*60)
    
    if args.skip_preprocessing:
        logger.info("Loading cached preprocessed data...")
        daily_df = pd.read_csv(f'{config.CACHE_DIR}/daily_aggregated_data.csv',
                              parse_dates=['DATE'])
    else:
        daily_df = preprocess_data(config)
    
    train_df, val_df, test_df, feature_engineer, sequences = engineer_features(
        daily_df, config
    )
    
    train_features, val_features, test_features = prepare_datasets(
        train_df, val_df, test_df, feature_engineer, sequences, config
    )
    
    results_df = train_models(train_features, val_features, test_features, config)
    
    logger.info("\n" + "="*60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"End time: {datetime.now()}")
    logger.info(f"Results saved to: {config.OUTPUT_DIR}")
    logger.info("="*60)
    
    return results_df


if __name__ == '__main__':
    main()
