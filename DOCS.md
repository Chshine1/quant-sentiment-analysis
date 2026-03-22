# Stock Return Prediction with Twitter Sentiment Analysis

A complete experimental pipeline for predicting stock returns using Twitter sentiment and financial data.

## Project Structure

```
quant-sentiment-analysis/
├── README.md
├── requirements.txt
├── main.py                    # Main execution script
├── src/
│   ├── __init__.py
│   ├── preprocessing_and_sentiment.py   # Module 1: Data preprocessing & sentiment
│   ├── feature_engineering.py           # Module 2: Feature construction
│   ├── training_pipeline.py             # Training & evaluation
│   └── models/
│       ├── __init__.py
│       └── lstm_attention_model.py      # LSTM+Attention architecture
├── data/
│   ├── tweets.csv           # Input data (18k rows)
│   └── cache/               # Cached intermediate results
├── output/                  # Model checkpoints & results
└── logs/                    # Pipeline logs
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python main.py --input data/tweets.csv --output output/
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input`, `-i` | Input CSV file path | `data/tweets.csv` |
| `--output`, `-o` | Output directory | `output/` |
| `--skip-preprocessing` | Skip preprocessing (use cached) | False |
| `--skip-textblob` | Skip TextBlob sentiment | False |
| `--skip-finbert` | Skip FinBERT sentiment | False |
| `--force-recompute` | Force recompute even if cached | False |
| `--epochs` | Number of training epochs | 100 |
| `--batch-size` | Batch size | 32 |
| `--lr` | Learning rate | 0.001 |

## Pipeline Overview

### Module 1: Preprocessing & Sentiment Computation

**File:** `src/preprocessing_and_sentiment.py`

**Features:**
- CSV loading with encoding handling (handles `utf-8`, `latin-1`, `cp1252`)
- Date parsing and standardization
- TextBlob sentiment computation (uses precomputed if available)
- FinBERT sentiment with caching mechanism
- Daily sentiment aggregation per company
- Control variable retrieval (optional yfinance integration)

**Key Classes:**
- `DataPreprocessor`: Main class for data preprocessing
- FinBERT caching: Saves to `data/cache/finbert_sentiment.csv`

### Module 2: Feature Engineering

**File:** `src/feature_engineering.py`

**Feature Groups:**

| Group | Dimensions | Description |
|-------|------------|-------------|
| Basic | 4 | Log volume, volatility (10D, 30D), normalized price |
| Count | 1 | Log tweet count |
| Sentiment | 4 | Mean/std FinBERT sentiment, positive/negative ratios |
| Temporal | 3 | Momentum (positive, negative, net) |
| Decay | 1 | Exponential decay-weighted sentiment (λ=0.3, T=7) |
| Control | 5 | S&P500 return, VIX, market cap, B/M, past 5D return |
| Calendar | 9 | Day-of-week (5) + Month (4) dummies |
| LSTM | 64 | Attention-weighted sequence encoding |

**Key Classes:**
- `FeatureEngineer`: Feature construction and scaling
- `DatasetBuilder`: PyTorch dataset creation

### LSTM+Attention Model

**File:** `src/models/lstm_attention_model.py`

**Architecture:**
```
Sentiment Sequence (T=7)
        ↓
   LSTM Encoder (hidden=64)
        ↓
Temporal Attention Layer
        ↓
  h_attn (64-dim)
        ↓
Feature Fusion (basic + count + sentiment + ... + h_attn)
        ↓
   FC Layers (128 → 64 → 32)
        ↓
   Regression Head → Return Prediction
```

### Training Pipeline

**File:** `src/training_pipeline.py`

**Configuration:**
- Optimizer: Adam (lr=0.001, weight_decay=1e-5)
- Loss: MSE
- Batch size: 32
- Early stopping: 5 epochs patience
- Chronological split: Train(Jul-Sep), Test(Oct)

**Evaluation Metrics:**
- MSE, RMSE, MAE, MAPE
- R²
- Directional Accuracy (% correct sign prediction)

## Input Data Format

Expected CSV columns:
```
TWEET, STOCK, DATE, LAST_PRICE, 
1_DAY_RETURN, 2_DAY_RETURN, 3_DAY_RETURN, 7_DAY_RETURN,
PX_VOLUME, VOLATILITY_10D, VOLATILITY_30D, 
LSTM_POLARITY, TEXTBLOB_POLARITY, MENTION, STOCK_CODE, cleaned_text
```

## Output

After training, the following files are generated in `output/`:

```
output/
├── model_1_DAY_RETURN.pt    # Model checkpoint
├── model_2_DAY_RETURN.pt
├── model_3_DAY_RETURN.pt
├── model_7_DAY_RETURN.pt
├── evaluation_results.csv   # Results table
└── results_summary.json      # Detailed results
```

## Example Results Table

| Horizon | MSE | RMSE | MAE | MAPE | R2 | Dir_Acc |
|---------|-----|------|-----|------|----|---------|
| 1_DAY_RETURN | X.XXXX | X.XXXX | X.XXXX | XX.XX | X.XXXX | XX.XX |
| 2_DAY_RETURN | X.XXXX | X.XXXX | X.XXXX | XX.XX | X.XXXX | XX.XX |
| 3_DAY_RETURN | X.XXXX | X.XXXX | X.XXXX | XX.XX | X.XXXX | XX.XX |
| 7_DAY_RETURN | X.XXXX | X.XXXX | X.XXXX | XX.XX | X.XXXX | XX.XX |

## Modularity

The code is organized into independent modules that can be used separately:

```python
from src.preprocessing_and_sentiment import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.training_pipeline import StockReturnTrainer

# Step 1: Preprocess
preprocessor = DataPreprocessor()
daily_df = preprocessor.process_full_pipeline('data/tweets.csv')

# Step 2: Engineer features
engineer = FeatureEngineer()
daily_df = engineer.build_all_features(daily_df)

# Step 3: Train
trainer = StockReturnTrainer(output_dir='output/')
results = trainer.train_all_horizons(train_data, val_data, test_data)
```

## Notes

1. **FinBERT Caching**: FinBERT inference is computationally expensive. Results are cached to `data/cache/finbert_sentiment.csv` to avoid recomputation.

2. **Control Variables**: Some control variables (MARKET_CAP, BOOK_TO_MARKET) may not be available. Placeholders are used; in practice, these should be fetched from financial data providers.

3. **GPU Support**: The code automatically uses GPU if available (CUDA). Training on CPU is supported but slower.

4. **Chronological Split**: The train/test split preserves temporal order to avoid data leakage.
