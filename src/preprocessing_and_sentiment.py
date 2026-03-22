"""
Module 1: Preprocessing & Sentiment Computation

This module handles:
- Loading and cleaning CSV data with proper encoding
- Computing TextBlob sentiment (if not precomputed)
- Computing FinBERT sentiment with caching mechanism
- Daily sentiment aggregation per company
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles data loading, cleaning, and sentiment computation."""
    
    REQUIRED_COLUMNS = [
        'TWEET', 'STOCK', 'DATE', 'LAST_PRICE', 
        '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN',
        'PX_VOLUME', 'VOLATILITY_10D', 'VOLATILITY_30D'
    ]
    
    OPTIONAL_COLUMNS = [
        'LSTM_POLARITY', 'TEXTBLOB_POLARITY', 'MENTION', 'STOCK_CODE', 'cleaned_text'
    ]
    
    CONTROL_COLUMNS = [
        'SP500_RETURN', 'VIX', 'MARKET_CAP', 'BOOK_TO_MARKET', 
        'PAST_5D_RETURN', 'DOW', 'MONTH'
    ]
    
    def __init__(self, data_dir: str = 'data', cache_dir: str = 'data/cache'):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.finbert_cache_path = self.cache_dir / 'finbert_sentiment.csv'
        self.processed_data_path = self.cache_dir / 'daily_aggregated_data.csv'
        
        self.finbert_model = None
        self.finbert_tokenizer = None
        self.cached_finbert_df = None
    
    def load_csv(self, filepath: str, encoding: str = 'utf-8', 
                 errors: str = 'replace') -> pd.DataFrame:
        """
        Load CSV with proper encoding handling.
        
        Args:
            filepath: Path to the CSV file
            encoding: Character encoding (default: utf-8)
            errors: How to handle encoding errors (default: replace)
        
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading CSV from {filepath}")
        
        try:
            df = pd.read_csv(filepath, encoding=encoding, errors=errors)
        except Exception as e:
            logger.warning(f"Failed with {encoding}, trying alternative encodings...")
            for alt_encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                try:
                    df = pd.read_csv(filepath, encoding=alt_encoding)
                    logger.info(f"Successfully loaded with {alt_encoding} encoding")
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(f"Could not load CSV with any encoding: {e}")
        
        logger.info(f"Loaded {len(df)} rows")
        
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            logger.warning(f"Missing required columns: {missing_cols}")
        
        df = self._standardize_columns(df)
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and create necessary mappings."""
        if 'STOCK_CODE' not in df.columns and 'STOCK' in df.columns:
            df['STOCK_CODE'] = df['STOCK']
        
        if 'cleaned_text' not in df.columns and 'TWEET' in df.columns:
            df['cleaned_text'] = df['TWEET']
        
        if 'TEXTBLOB_POLARITY' not in df.columns:
            df['TEXTBLOB_POLARITY'] = np.nan
        
        return df
    
    def parse_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse DATE column to datetime and set as index.
        
        DATE format: YYYY/M/D (e.g., 2017/3/31 or 2018/10/15)
        """
        logger.info("Parsing dates...")
        
        df['DATE'] = pd.to_datetime(df['DATE'], format='mixed', dayfirst=False)
        df = df.sort_values(['STOCK_CODE', 'DATE']).reset_index(drop=True)
        
        logger.info(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")
        
        return df
    
    def compute_textblob_sentiment(self, df: pd.DataFrame, 
                                   text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Compute TextBlob sentiment polarity for tweets.
        
        Uses precomputed TEXTBLOB_POLARITY if available.
        """
        if TextBlob is None:
            logger.warning("TextBlob not available. Install with: pip install textblob")
            return df
        
        mask = df['TEXTBLOB_POLARITY'].isna()
        missing_count = mask.sum()
        
        if missing_count == 0:
            logger.info("TextBlob polarity already computed for all tweets")
            return df
        
        logger.info(f"Computing TextBlob sentiment for {missing_count} missing tweets...")
        
        for idx in tqdm(df[mask].index, desc="TextBlob"):
            text = str(df.loc[idx, text_column])
            sentiment = TextBlob(text).sentiment.polarity
            df.loc[idx, 'TEXTBLOB_POLARITY'] = sentiment
        
        return df
    
    def _init_finbert(self):
        """Initialize FinBERT model and tokenizer."""
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers library not available. Install with: pip install transformers")
        
        if self.finbert_model is None:
            logger.info("Loading FinBERT model (ProsusAI/finbert)...")
            self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            self.finbert_model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
            self.finbert_model.to(self.device)
            self.finbert_model.eval()
            logger.info("FinBERT model loaded successfully")
    
    def _get_tweet_hash(self, text: str) -> str:
        """Generate hash for tweet text to use as identifier."""
        return hashlib.md5(str(text).encode()).hexdigest()
    
    def _load_finbert_cache(self) -> pd.DataFrame:
        """Load cached FinBERT results if available."""
        if self.cached_finbert_df is not None:
            return self.cached_finbert_df
        
        if self.finbert_cache_path.exists():
            self.cached_finbert_df = pd.read_csv(self.finbert_cache_path)
            logger.info(f"Loaded {len(self.cached_finbert_df)} cached FinBERT results")
        else:
            self.cached_finbert_df = pd.DataFrame(
                columns=['tweet_hash', 'STOCK_CODE', 'DATE', 'finbert_polarity']
            )
        
        return self.cached_finbert_df
    
    def _save_finbert_cache(self, df: pd.DataFrame):
        """Save FinBERT results to cache."""
        df.to_csv(self.finbert_cache_path, index=False)
        logger.info(f"Saved FinBERT cache to {self.finbert_cache_path}")
    
    def compute_finbert_sentiment(self, df: pd.DataFrame, 
                                  text_column: str = 'cleaned_text',
                                  batch_size: int = 32) -> pd.DataFrame:
        """
        Compute FinBERT sentiment for tweets with caching.
        
        FinBERT returns sentiment probabilities for positive, negative, neutral.
        Sentiment score = positive_prob - negative_prob (range: -1 to 1)
        
        Args:
            df: DataFrame with tweets
            text_column: Column containing tweet text
            batch_size: Batch size for inference
        
        Returns:
            DataFrame with added finbert_polarity column
        """
        self._init_finbert()
        cache_df = self._load_finbert_cache()
        
        df['tweet_hash'] = df[text_column].apply(self._get_tweet_hash)
        df['finbert_polarity'] = np.nan
        
        existing_hashes = set(cache_df['tweet_hash'].values)
        df_indices_to_compute = []
        
        for idx in df.index:
            tweet_hash = df.loc[idx, 'tweet_hash']
            if tweet_hash in existing_hashes:
                cached_result = cache_df[cache_df['tweet_hash'] == tweet_hash]
                if not cached_result.empty:
                    df.loc[idx, 'finbert_polarity'] = cached_result.iloc[0]['finbert_polarity']
            else:
                df_indices_to_compute.append(idx)
        
        if not df_indices_to_compute:
            logger.info("All tweets already have FinBERT sentiment (from cache)")
            return df
        
        logger.info(f"Computing FinBERT sentiment for {len(df_indices_to_compute)} new tweets...")
        
        new_results = []
        
        for i in tqdm(range(0, len(df_indices_to_compute), batch_size), desc="FinBERT"):
            batch_indices = df_indices_to_compute[i:i+batch_size]
            batch_texts = [str(df.loc[idx, text_column]) for idx in batch_indices]
            
            inputs = self.finbert_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.finbert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
            
            sentiment_scores = probs[:, 0].cpu().numpy() - probs[:, 1].cpu().numpy()
            
            for j, idx in enumerate(batch_indices):
                df.loc[idx, 'finbert_polarity'] = sentiment_scores[j]
                new_results.append({
                    'tweet_hash': df.loc[idx, 'tweet_hash'],
                    'STOCK_CODE': df.loc[idx, 'STOCK_CODE'],
                    'DATE': df.loc[idx, 'DATE'],
                    'finbert_polarity': sentiment_scores[j]
                })
            
            if (i // batch_size + 1) % 50 == 0:
                temp_df = pd.DataFrame(new_results)
                if not cache_df.empty:
                    temp_df = pd.concat([cache_df, temp_df], ignore_index=True)
                self._save_finbert_cache(temp_df)
        
        if new_results:
            final_df = pd.concat([cache_df, pd.DataFrame(new_results)], ignore_index=True)
            self._save_finbert_cache(final_df)
        
        return df
    
    def aggregate_daily_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate tweet-level sentiment to daily level per company.
        
        Creates daily_aggregated_data.csv with columns:
        - DATE, STOCK_CODE
        - daily_sentiment_textblob, daily_sentiment_finbert
        - tweet_count
        - Financial features: PX_VOLUME, VOLATILITY_10D, VOLATILITY_30D, LAST_PRICE
        - Return features: 1_DAY_RETURN, 2_DAY_RETURN, 3_DAY_RETURN, 7_DAY_RETURN
        """
        logger.info("Aggregating daily sentiment...")
        
        agg_dict = {
            'TEXTBLOB_POLARITY': ['mean', 'std'],
            'finbert_polarity': ['mean', 'std'],
            'TWEET': 'count',
            'PX_VOLUME': 'first',
            'VOLATILITY_10D': 'first',
            'VOLATILITY_30D': 'first',
            'LAST_PRICE': 'first',
            '1_DAY_RETURN': 'first',
            '2_DAY_RETURN': 'first',
            '3_DAY_RETURN': 'first',
            '7_DAY_RETURN': 'first'
        }
        
        daily_df = df.groupby(['DATE', 'STOCK_CODE']).agg(agg_dict).reset_index()
        
        daily_df.columns = [
            'DATE', 'STOCK_CODE',
            'daily_sentiment_textblob', 'textblob_std',
            'daily_sentiment_finbert', 'finbert_std',
            'tweet_count',
            'PX_VOLUME', 'VOLATILITY_10D', 'VOLATILITY_30D',
            'LAST_PRICE', '1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN'
        ]
        
        daily_df['finbert_std'] = daily_df['finbert_std'].fillna(0)
        daily_df['textblob_std'] = daily_df['textblob_std'].fillna(0)
        
        daily_df = daily_df.sort_values(['STOCK_CODE', 'DATE']).reset_index(drop=True)
        
        logger.info(f"Created daily aggregation with {len(daily_df)} company-day observations")
        
        return daily_df
    
    def add_control_variables(self, df: pd.DataFrame,
                             use_yfinance: bool = True) -> pd.DataFrame:
        """
        Add control variables to the dataset.
        
        If available from CSV, uses existing values.
        Otherwise, attempts to fetch from yfinance or uses placeholders.
        """
        logger.info("Adding control variables...")
        
        for col in ['SP500_RETURN', 'VIX', 'MARKET_CAP', 'BOOK_TO_MARKET', 
                    'PAST_5D_RETURN', 'DOW', 'MONTH']:
            if col not in df.columns:
                df[col] = np.nan
        
        df['DOW'] = df['DATE'].dt.dayofweek
        df['MONTH'] = df['DATE'].dt.month
        
        if use_yfinance:
            try:
                import yfinance as yf
                self._fetch_market_data(df, yf)
            except ImportError:
                logger.warning("yfinance not available. Market control variables will use placeholders.")
        
        df = self._compute_past_return(df)
        df = self._fill_control_placeholders(df)
        
        return df
    
    def _fetch_market_data(self, df: pd.DataFrame, yf):
        """Fetch S&P 500 and VIX data from yfinance."""
        logger.info("Fetching market data from yfinance...")
        
        date_min = df['DATE'].min() - pd.Timedelta(days=10)
        date_max = df['DATE'].max()
        
        try:
            sp500 = yf.download('^GSPC', start=date_min, end=date_max, progress=False)
            vix = yf.download('^VIX', start=date_min, end=date_max, progress=False)
            
            sp500['SP500_RETURN'] = sp500['Close'].pct_change() * 100
            sp500.index = pd.to_datetime(sp500.index).normalize()
            vix.index = pd.to_datetime(vix.index).normalize()
            
            df['DATE_normalized'] = df['DATE'].dt.normalize()
            
            sp500_reset = sp500[['SP500_RETURN']].reset_index()
            sp500_reset.columns = ['DATE_normalized', 'SP500_RETURN']
            
            vix_reset = vix[['Close']].reset_index()
            vix_reset.columns = ['DATE_normalized', 'VIX']
            
            df = df.merge(sp500_reset, on='DATE_normalized', how='left')
            df = df.merge(vix_reset, on='DATE_normalized', how='left')
            df = df.drop('DATE_normalized', axis=1)
            
            logger.info("Market data fetched successfully")
        except Exception as e:
            logger.warning(f"Failed to fetch market data: {e}")
    
    def _compute_past_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute past 5-day return for each stock."""
        df = df.sort_values(['STOCK_CODE', 'DATE'])
        
        df['PAST_5D_RETURN'] = df.groupby('STOCK_CODE')['1_DAY_RETURN'].transform(
            lambda x: x.shift(1).rolling(window=5, min_periods=5).sum()
        )
        
        return df
    
    def _fill_control_placeholders(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing control variables with reasonable defaults."""
        if df['MARKET_CAP'].isna().any():
            df['MARKET_CAP'] = df['LAST_PRICE'] * df['PX_VOLUME']
        
        if df['VIX'].isna().any():
            df['VIX'] = 15.0
        
        if df['SP500_RETURN'].isna().any():
            df['SP500_RETURN'] = 0.0
        
        if df['BOOK_TO_MARKET'].isna().any():
            df['BOOK_TO_MARKET'] = 0.5
        
        return df
    
    def process_full_pipeline(self, csv_path: str,
                             compute_textblob: bool = True,
                             compute_finbert: bool = True,
                             force_recompute: bool = False) -> pd.DataFrame:
        """
        Run the full preprocessing pipeline.
        
        Args:
            csv_path: Path to the input CSV file
            compute_textblob: Whether to compute TextBlob sentiment
            compute_finbert: Whether to compute FinBERT sentiment
            force_recompute: Force recompute even if cached
        
        Returns:
            Processed daily aggregated DataFrame
        """
        if self.processed_data_path.exists() and not force_recompute:
            logger.info("Loading cached processed data...")
            return pd.read_csv(self.processed_data_path, parse_dates=['DATE'])
        
        df = self.load_csv(csv_path)
        df = self.parse_dates(df)
        
        if compute_textblob:
            df = self.compute_textblob_sentiment(df)
        
        if compute_finbert:
            df = self.compute_finbert_sentiment(df)
        
        daily_df = self.aggregate_daily_sentiment(df)
        daily_df = self.add_control_variables(daily_df)
        
        self.processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        daily_df.to_csv(self.processed_data_path, index=False)
        logger.info(f"Saved processed data to {self.processed_data_path}")
        
        return daily_df


def main():
    """Example usage of the preprocessing pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess Twitter sentiment data')
    parser.add_argument('--input', '-i', required=True, help='Input CSV path')
    parser.add_argument('--output', '-o', default='data/cache', help='Output cache directory')
    parser.add_argument('--skip-textblob', action='store_true', help='Skip TextBlob computation')
    parser.add_argument('--skip-finbert', action='store_true', help='Skip FinBERT computation')
    parser.add_argument('--force', '-f', action='store_true', help='Force recompute')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(cache_dir=args.output)
    
    daily_df = preprocessor.process_full_pipeline(
        csv_path=args.input,
        compute_textblob=not args.skip_textblob,
        compute_finbert=not args.skip_finbert,
        force_recompute=args.force
    )
    
    logger.info(f"Processing complete! Daily data shape: {daily_df.shape}")
    logger.info(f"Date range: {daily_df['DATE'].min()} to {daily_df['DATE'].max()}")
    logger.info(f"Unique stocks: {daily_df['STOCK_CODE'].nunique()}")


if __name__ == '__main__':
    main()
