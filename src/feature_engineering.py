"""
Module 2: Feature Construction

This module handles:
- Feature engineering (basic, count, sentiment, temporal, control variables)
- Creating feature sequences for LSTM+Attention model
- Train/test chronological splits
- Data normalization
"""

import logging
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Handles feature engineering for stock return prediction."""
    
    T = 7
    LAMBDA = 0.3
    
    def __init__(self):
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self._fitted = False
    
    def build_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build basic financial features (4 dimensions):
        - PX_VOLUME (log-transformed)
        - VOLATILITY_10D
        - VOLATILITY_30D
        - LAST_PRICE (normalized)
        """
        df['log_volume'] = np.log1p(df['PX_VOLUME'])
        df['norm_price'] = df['LAST_PRICE']
        
        return df
    
    def build_count_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build count features (2 dimensions):
        - log(tweet_count)
        """
        df['log_tweet_count'] = np.log1p(df['tweet_count'])
        
        return df
    
    def build_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build sentiment features over past T days (4 dimensions):
        - Mean of daily FinBERT sentiment
        - Std of daily FinBERT sentiment
        - Proportion of positive days (sentiment > 0.1)
        - Proportion of negative days (sentiment < -0.1)
        """
        df = df.sort_values(['STOCK_CODE', 'DATE'])
        
        sentiment_mean = df.groupby('STOCK_CODE')['daily_sentiment_finbert'].transform(
            lambda x: x.shift(1).rolling(window=self.T, min_periods=1).mean()
        )
        df['sentiment_mean_T'] = sentiment_mean
        
        sentiment_std = df.groupby('STOCK_CODE')['daily_sentiment_finbert'].transform(
            lambda x: x.shift(1).rolling(window=self.T, min_periods=2).std()
        )
        df['sentiment_std_T'] = sentiment_std.fillna(0)
        
        df['positive_ratio'] = df.groupby('STOCK_CODE')['daily_sentiment_finbert'].transform(
            lambda x: x.shift(1).rolling(window=self.T, min_periods=1).apply(
                lambda y: (y > 0.1).mean(), raw=False
            )
        )
        
        df['negative_ratio'] = df.groupby('STOCK_CODE')['daily_sentiment_finbert'].transform(
            lambda x: x.shift(1).rolling(window=self.T, min_periods=1).apply(
                lambda y: (y < -0.1).mean(), raw=False
            )
        )
        
        return df
    
    def build_temporal_sentiment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build temporal sentiment features (3 dimensions):
        - Sentiment momentum (positive, negative, net)
        - Decay-weighted sentiment (lambda=0.3)
        """
        df = df.sort_values(['STOCK_CODE', 'DATE'])
        
        df['sentiment_positive_momentum'] = df.groupby('STOCK_CODE')['daily_sentiment_finbert'].transform(
            lambda x: x.shift(1).rolling(window=self.T, min_periods=1).apply(
                lambda y: (y > 0.1).sum(), raw=True
            )
        )
        
        df['sentiment_negative_momentum'] = df.groupby('STOCK_CODE')['daily_sentiment_finbert'].transform(
            lambda x: x.shift(1).rolling(window=self.T, min_periods=1).apply(
                lambda y: (y < -0.1).sum(), raw=True
            )
        )
        
        df['sentiment_net_momentum'] = (
            df['sentiment_positive_momentum'] - df['sentiment_negative_momentum']
        )
        
        df['decay_weighted_sentiment'] = df.groupby('STOCK_CODE')['daily_sentiment_finbert'].transform(
            lambda x: x.shift(1).rolling(window=self.T, min_periods=1).apply(
                lambda y: self._compute_decay_weighted(y), raw=False
            )
        )
        
        return df
    
    def _compute_decay_weighted(self, values: np.ndarray) -> float:
        """Compute exponentially weighted sentiment with decay factor lambda."""
        if len(values) == 0:
            return 0.0
        
        weights = np.array([self.LAMBDA ** (len(values) - i - 1) for i in range(len(values))])
        weights = weights / weights.sum() if weights.sum() > 0 else np.ones_like(weights)
        
        return np.sum(values * weights)
    
    def build_control_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build control variable features."""
        for col in ['SP500_RETURN', 'VIX', 'MARKET_CAP', 'BOOK_TO_MARKET', 'PAST_5D_RETURN']:
            if col not in df.columns:
                df[col] = 0.0
        
        df['log_market_cap'] = np.log1p(df['MARKET_CAP'].fillna(0))
        
        return df
    
    def build_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build all features for the model."""
        logger.info("Building all features...")
        
        df = self.build_basic_features(df)
        df = self.build_count_features(df)
        df = self.build_sentiment_features(df)
        df = self.build_temporal_sentiment_features(df)
        df = self.build_control_features(df)
        
        logger.info("Feature engineering complete")
        
        return df
    
    def get_sentiment_sequences(self, df: pd.DataFrame, T: int = None) -> np.ndarray:
        """
        Extract sentiment sequences for LSTM input.
        
        For each company-day, extract the last T days of daily FinBERT sentiment.
        
        Returns:
            Array of shape (n_samples, T)
        """
        if T is None:
            T = self.T
        
        df = df.sort_values(['STOCK_CODE', 'DATE'])
        
        sequences = []
        for stock in df['STOCK_CODE'].unique():
            stock_df = df[df['STOCK_CODE'] == stock].sort_values('DATE')
            sentiments = stock_df['daily_sentiment_finbert'].values
            
            for i in range(len(sentiments)):
                start_idx = max(0, i - T)
                seq = sentiments[start_idx:i]
                
                if len(seq) < T:
                    padding = np.zeros(T - len(seq))
                    seq = np.concatenate([padding, seq])
                
                sequences.append(seq)
        
        result_df = df.sort_values(['STOCK_CODE', 'DATE']).reset_index(drop=True)
        result = np.array(sequences)
        
        assert len(result) == len(result_df), "Sequence length mismatch"
        
        return result
    
    def create_temporal_split(self, df: pd.DataFrame,
                             train_start: str = '2018-07-01',
                             train_end: str = '2018-09-15',
                             test_start: str = '2018-10-01',
                             test_end: str = '2018-10-31',
                             val_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create chronological train/test/validation splits.
        
        Training: July 1, 2018 – September 15, 2018
        Validation: Last 20% of training period
        Test: October 1, 2018 – October 31, 2018
        """
        df = df.copy()
        df['DATE'] = pd.to_datetime(df['DATE'])
        
        train_mask = (df['DATE'] >= train_start) & (df['DATE'] <= train_end)
        test_mask = (df['DATE'] >= test_start) & (df['DATE'] <= test_end)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        train_df = train_df.sort_values('DATE')
        split_idx = int(len(train_df) * (1 - val_ratio))
        
        train_actual = train_df.iloc[:split_idx]
        val_df = train_df.iloc[split_idx:]
        
        logger.info(f"Train: {len(train_actual)} samples, Val: {len(val_df)} samples, Test: {len(test_df)} samples")
        
        return train_actual, val_df, test_df
    
    def fit_scalers(self, train_df: pd.DataFrame):
        """Fit scalers on training data only."""
        self._fitted = True
        
        basic_cols = ['log_volume', 'VOLATILITY_10D', 'VOLATILITY_30D', 'norm_price']
        self.scalers['basic'] = StandardScaler()
        self.scalers['basic'].fit(train_df[basic_cols].fillna(0))
        
        count_cols = ['log_tweet_count']
        self.scalers['count'] = StandardScaler()
        self.scalers['count'].fit(train_df[count_cols].fillna(0))
        
        sentiment_cols = ['sentiment_mean_T', 'sentiment_std_T', 'positive_ratio', 'negative_ratio']
        self.scalers['sentiment'] = StandardScaler()
        self.scalers['sentiment'].fit(train_df[sentiment_cols].fillna(0))
        
        control_cols = ['SP500_RETURN', 'VIX', 'log_market_cap', 'BOOK_TO_MARKET', 'PAST_5D_RETURN']
        self.scalers['control'] = StandardScaler()
        self.scalers['control'].fit(train_df[control_cols].fillna(0))
        
        logger.info("Scalers fitted on training data")
    
    def transform_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform features using fitted scalers.
        
        Returns:
            Tuple of (basic_features, scaled_features_dict)
        """
        if not self._fitted:
            raise RuntimeError("Scalers not fitted. Call fit_scalers first.")
        
        basic_cols = ['log_volume', 'VOLATILITY_10D', 'VOLATILITY_30D', 'norm_price']
        basic = self.scalers['basic'].transform(df[basic_cols].fillna(0))
        
        count_cols = ['log_tweet_count']
        count = self.scalers['count'].transform(df[count_cols].fillna(0))
        
        sentiment_cols = ['sentiment_mean_T', 'sentiment_std_T', 'positive_ratio', 'negative_ratio']
        sentiment = self.scalers['sentiment'].transform(df[sentiment_cols].fillna(0))
        
        momentum_cols = ['sentiment_positive_momentum', 'sentiment_negative_momentum', 'sentiment_net_momentum']
        momentum = df[momentum_cols].fillna(0).values
        
        decay_cols = ['decay_weighted_sentiment']
        decay = df[decay_cols].fillna(0).values
        
        control_cols = ['SP500_RETURN', 'VIX', 'log_market_cap', 'BOOK_TO_MARKET', 'PAST_5D_RETURN']
        control = self.scalers['control'].transform(df[control_cols].fillna(0))
        
        dow_cols = ['DOW']
        dow_dummies = pd.get_dummies(df['DOW'], prefix='dow')
        for d in range(7):
            if d not in dow_dummies.columns:
                dow_dummies[d] = 0
        dow = dow_dummies[[i for i in range(7)]].values
        
        month_cols = ['MONTH']
        month_dummies = pd.get_dummies(df['MONTH'], prefix='month')
        for m in [7, 8, 9, 10]:
            if m not in month_dummies.columns:
                month_dummies[m] = 0
        month = month_dummies[[7, 8, 9, 10]].values
        
        return basic, {
            'count': count,
            'sentiment': sentiment,
            'momentum': momentum,
            'decay': decay,
            'control': control,
            'dow': dow,
            'month': month
        }
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Return dimensions of each feature group."""
        return {
            'basic': 4,
            'count': 1,
            'sentiment': 4,
            'momentum': 3,
            'decay': 1,
            'control': 5,
            'dow': 7,
            'month': 4,
            'lstm_hidden': 64
        }


class DatasetBuilder:
    """Builds PyTorch datasets for training."""
    
    def __init__(self, feature_engineer: FeatureEngineer):
        self.feature_engineer = feature_engineer
    
    def build_dataset(self, df: pd.DataFrame, 
                     target_col: str,
                     sequences: np.ndarray = None) -> Dict[str, np.ndarray]:
        """
        Build complete dataset for model training.
        
        Args:
            df: DataFrame with features
            target_col: Target column name (e.g., '1_DAY_RETURN')
            sequences: Pre-computed sentiment sequences
        
        Returns:
            Dictionary with feature arrays
        """
        df = df.copy()
        df = df.reset_index(drop=True)
        
        basic, other_features = self.feature_engineer.transform_features(df)
        
        if sequences is None:
            sequences = self.feature_engineer.get_sentiment_sequences(df)
        
        n_samples = len(df)
        
        if len(sequences) != n_samples:
            raise ValueError(f"Sequence length {len(sequences)} != sample length {n_samples}")
        
        features_dict = {
            'basic': basic,
            'count': other_features['count'],
            'sentiment': other_features['sentiment'],
            'momentum': other_features['momentum'],
            'decay': other_features['decay'],
            'control': other_features['control'],
            'dow': other_features['dow'],
            'month': other_features['month'],
            'sequences': sequences,
            'targets': df[target_col].values.reshape(-1, 1),
            'stock_codes': df['STOCK_CODE'].values,
            'dates': df['DATE'].values
        }
        
        return features_dict


def create_lookback_data(df: pd.DataFrame, feature_engineer: FeatureEngineer,
                         target_col: str = '1_DAY_RETURN',
                         T: int = 7) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create data with sufficient lookback period for feature computation.
    
    To compute T-day features, we need data from T days before the start date.
    """
    min_date = df['DATE'].min()
    cutoff_date = min_date + pd.Timedelta(days=T)
    
    logger.info(f"Dropping first {T} days for lookback period. Cutoff: {cutoff_date}")
    
    df_filtered = df[df['DATE'] >= cutoff_date].copy()
    df_filtered = df_filtered.sort_values(['STOCK_CODE', 'DATE']).reset_index(drop=True)
    sequences = feature_engineer.get_sentiment_sequences(df_filtered, T=T)
    
    return df_filtered, sequences


def main():
    """Example usage of feature engineering."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Feature engineering for stock prediction')
    parser.add_argument('--input', '-i', required=True, help='Input CSV path (daily aggregated)')
    parser.add_argument('--output', '-o', default='data/features', help='Output directory')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.input, parse_dates=['DATE'])
    
    engineer = FeatureEngineer()
    df = engineer.build_all_features(df)
    
    train_df, val_df, test_df = engineer.create_temporal_split(df)
    
    engineer.fit_scalers(train_df)
    
    logger.info(f"Processed data saved to {args.output}")


if __name__ == '__main__':
    main()
