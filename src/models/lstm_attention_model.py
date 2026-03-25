"""
LSTM+Attention Model Architecture

Implements:
- LSTM encoder for sentiment sequences
- Temporal attention layer
- Feature fusion network
- Regression head for stock return prediction
"""

import logging
from typing import Dict, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class SentimentLSTMEncoder(nn.Module):
    """
    LSTM encoder for sentiment sequences.
    
    Input: (batch_size, T) - T daily sentiment scores
    Output: (batch_size, hidden_size) - encoded sequence representation
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, num_layers: int = 1, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) sentiment values
        
        Returns:
            hidden_state: (batch_size, hidden_size)
        """
        x = x.unsqueeze(-1)
        
        output, (hidden, cell) = self.lstm(x)
        
        return hidden[-1]


class TemporalAttentionLayer(nn.Module):
    """
    Temporal attention mechanism over LSTM hidden states.
    
    Computes attention weights over time steps and returns weighted sum.
    """
    
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        
        self.attention_weights = nn.Linear(hidden_size, 1, bias=False)
    
    def forward(self, lstm_outputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            lstm_outputs: (batch_size, seq_len, hidden_size)
        
        Returns:
            context_vector: (batch_size, hidden_size) - attention-weighted representation
            attention_weights: (batch_size, seq_len) - attention scores
        """
        scores = self.attention_weights(lstm_outputs)
        scores = scores.squeeze(-1)
        
        attention_weights = F.softmax(scores, dim=-1)
        
        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            lstm_outputs
        ).squeeze(1)
        
        return context_vector, attention_weights


class SentimentEncoderWithAttention(nn.Module):
    """
    LSTM encoder with temporal attention for sentiment sequences.
    
    Returns both the final hidden state and attention-weighted representation.
    """
    
    def __init__(self, input_size: int = 1, hidden_size: int = 64, 
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.attention = TemporalAttentionLayer(hidden_size)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len)
        
        Returns:
            h_seq: (batch_size, hidden_size) - final hidden state
            h_attn: (batch_size, hidden_size) - attention-weighted representation
            attention_weights: (batch_size, seq_len)
        """
        x = x.unsqueeze(-1)
        
        lstm_out, _ = self.lstm(x)
        
        h_attn, attention_weights = self.attention(lstm_out)
        
        h_seq = lstm_out[:, -1, :]
        
        return h_seq, h_attn, attention_weights


class StockReturnPredictionModel(nn.Module):
    """
    Complete model for stock return prediction.
    
    Architecture:
    1. LSTM+Attention encoder for sentiment sequences -> h_seq (64-dim)
    2. Feature fusion layer (combining all feature groups)
    3. Fully connected layers for regression
    """
    
    def __init__(self,
                 seq_len: int = 7,
                 lstm_hidden_size: int = 64,
                 lstm_num_layers: int = 1,
                 lstm_dropout: float = 0.1,
                 feature_dims: Optional[Dict[str, int]] = None,
                 hidden_dims: Tuple[int, ...] = (128, 64, 32),
                 dropout: float = 0.3):
        
        super().__init__()
        
        if feature_dims is None:
            feature_dims = {
                'basic': 4,
                'count': 1,
                'sentiment': 4,
                'momentum': 3,
                'decay': 1,
                'control': 5,
                'dow': 7,
                'month': 4
            }
        
        self.feature_dims = feature_dims
        total_tabular = sum(feature_dims.values())
        total_features = total_tabular + lstm_hidden_size
        
        self.sentiment_encoder = SentimentEncoderWithAttention(
            input_size=1,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            dropout=lstm_dropout
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_features, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        layers = []
        for i in range(len(hidden_dims) - 1):
            layers.extend([
                nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                nn.BatchNorm1d(hidden_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        
        self.regression_head = nn.Sequential(*layers)
        
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, sentiment_seq: torch.Tensor,
               basic_features: torch.Tensor,
               count_features: torch.Tensor,
               sentiment_features: torch.Tensor,
               momentum_features: torch.Tensor,
               decay_features: torch.Tensor,
               control_features: torch.Tensor,
               dow_features: torch.Tensor,
               month_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            sentiment_seq: (batch_size, seq_len=7) sentiment sequence
            basic_features: (batch_size, 4)
            count_features: (batch_size, 1)
            sentiment_features: (batch_size, 4)
            momentum_features: (batch_size, 3)
            decay_features: (batch_size, 1)
            control_features: (batch_size, 5)
            dow_features: (batch_size, 5)
            month_features: (batch_size, 4)
        
        Returns:
            predictions: (batch_size, 1) predicted return
        """
        h_seq, h_attn, _ = self.sentiment_encoder(sentiment_seq)
        
        h_combined = torch.cat([
            basic_features,
            count_features,
            sentiment_features,
            momentum_features,
            decay_features,
            control_features,
            dow_features,
            month_features,
            h_attn
        ], dim=1)
        
        x = self.feature_fusion(h_combined)
        x = self.regression_head(x)
        x = self.output_layer(x)
        
        return x


class StockReturnDataset(Dataset):
    """PyTorch Dataset for stock return prediction."""
    
    def __init__(self, features_dict: Dict[str, np.ndarray]):
        self.features_dict = features_dict
        self.n_samples = len(features_dict['targets'])
        
        self._validate()
    
    def _validate(self):
        """Validate all feature arrays have the same length."""
        expected_len = len(self.features_dict['targets'])
        for key, arr in self.features_dict.items():
            if isinstance(arr, np.ndarray) and arr.ndim > 0:
                if len(arr) != expected_len:
                    raise ValueError(f"Feature {key} has length {len(arr)}, expected {expected_len}")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'sentiment_seq': torch.FloatTensor(self.features_dict['sequences'][idx]),
            'basic': torch.FloatTensor(self.features_dict['basic'][idx]),
            'count': torch.FloatTensor(self.features_dict['count'][idx]),
            'sentiment': torch.FloatTensor(self.features_dict['sentiment'][idx]),
            'momentum': torch.FloatTensor(self.features_dict['momentum'][idx]),
            'decay': torch.FloatTensor(self.features_dict['decay'][idx]),
            'control': torch.FloatTensor(self.features_dict['control'][idx]),
            'dow': torch.FloatTensor(self.features_dict['dow'][idx]),
            'month': torch.FloatTensor(self.features_dict['month'][idx]),
            'target': torch.FloatTensor(self.features_dict['targets'][idx]),
            'stock_code': self.features_dict['stock_codes'][idx],
            'date': self.features_dict['dates'][idx]
        }


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    return {
        'sentiment_seq': torch.stack([x['sentiment_seq'] for x in batch]),
        'basic': torch.stack([x['basic'] for x in batch]),
        'count': torch.stack([x['count'] for x in batch]),
        'sentiment': torch.stack([x['sentiment'] for x in batch]),
        'momentum': torch.stack([x['momentum'] for x in batch]),
        'decay': torch.stack([x['decay'] for x in batch]),
        'control': torch.stack([x['control'] for x in batch]),
        'dow': torch.stack([x['dow'] for x in batch]),
        'month': torch.stack([x['month'] for x in batch]),
        'target': torch.stack([x['target'] for x in batch]),
        'stock_code': [x['stock_code'] for x in batch],
        'date': [x['date'] for x in batch]
    }


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: StockReturnPredictionModel) -> str:
    """Get a summary of model architecture and parameters."""
    total_params = count_parameters(model)
    
    summary = f"Model Architecture:\n"
    summary += f"  Total trainable parameters: {total_params:,}\n"
    summary += f"\nSentiment Encoder:\n"
    summary += f"  LSTM hidden size: {model.sentiment_encoder.hidden_size}\n"
    summary += f"  LSTM layers: {model.sentiment_encoder.lstm.num_layers}\n"
    summary += f"\nFeature Dimensions:\n"
    for name, dim in model.feature_dims.items():
        summary += f"  {name}: {dim}\n"
    
    return summary
