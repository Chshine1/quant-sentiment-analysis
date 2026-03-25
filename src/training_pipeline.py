"""
Training and Evaluation Pipeline

Handles:
- Model training with early stopping
- Evaluation with regression metrics and directional accuracy
- Cross-horizon training (k=1, 2, 3, 7)
- Results logging and model checkpointing
"""

import os
import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm

from src.models.lstm_attention_model import (
    StockReturnPredictionModel,
    StockReturnDataset,
    collate_fn,
    count_parameters,
    get_model_summary
)

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate regression and directional accuracy metrics."""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(MetricsCalculator.mse(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.abs(y_true - y_pred))
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        mask = np.abs(y_true) > 1e-6
        if mask.sum() == 0:
            return 0.0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        if ss_tot < 1e-10:
            return 0.0
        return 1 - (ss_res / ss_tot)
    
    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Percentage of predictions where sign matches true return.
        """
        correct = np.sum(np.sign(y_true) == np.sign(y_pred))
        return (correct / len(y_true)) * 100
    
    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics."""
        return {
            'MSE': MetricsCalculator.mse(y_true, y_pred),
            'RMSE': MetricsCalculator.rmse(y_true, y_pred),
            'MAE': MetricsCalculator.mae(y_true, y_pred),
            'MAPE': MetricsCalculator.mape(y_true, y_pred),
            'R2': MetricsCalculator.r2(y_true, y_pred),
            'Directional_Accuracy': MetricsCalculator.directional_accuracy(y_true, y_pred)
        }


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_state = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False
    
    def reset(self):
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.best_state = None


class Trainer:
    """Handles model training."""
    
    def __init__(self,
                 model: StockReturnPredictionModel,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 device: Optional[torch.device] = None):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )
        
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            sentiment_seq = batch['sentiment_seq'].to(self.device)
            basic = batch['basic'].to(self.device)
            count = batch['count'].to(self.device)
            sentiment = batch['sentiment'].to(self.device)
            momentum = batch['momentum'].to(self.device)
            decay = batch['decay'].to(self.device)
            control = batch['control'].to(self.device)
            dow = batch['dow'].to(self.device)
            month = batch['month'].to(self.device)
            targets = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            
            outputs = self.model(
                sentiment_seq, basic, count, sentiment,
                momentum, decay, control, dow, month
            )
            
            loss = self.criterion(outputs, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / n_batches
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sentiment_seq = batch['sentiment_seq'].to(self.device)
                basic = batch['basic'].to(self.device)
                count = batch['count'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)
                momentum = batch['momentum'].to(self.device)
                decay = batch['decay'].to(self.device)
                control = batch['control'].to(self.device)
                dow = batch['dow'].to(self.device)
                month = batch['month'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(
                    sentiment_seq, basic, count, sentiment,
                    momentum, decay, control, dow, month
                )
                
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                n_batches += 1
        
        return total_loss / n_batches
    
    def train(self,
             train_loader: DataLoader,
             val_loader: DataLoader,
             n_epochs: int = 100,
             early_stopping_patience: int = 5,
             verbose: bool = True) -> Dict[str, List[float]]:
        """
        Train the model with early stopping.
        
        Returns:
            Dictionary with training history
        """
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
                )
            
            if early_stopping(val_loss, self.model):
                if verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if early_stopping.best_state is not None:
            self.model.load_state_dict(early_stopping.best_state)
        
        return {
            'train_loss': self.train_losses,
            'val_loss': self.val_losses
        }


class Evaluator:
    """Handles model evaluation."""
    
    def __init__(self, model: nn.Module, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions.
        
        Returns:
            Tuple of (predictions, targets)
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                sentiment_seq = batch['sentiment_seq'].to(self.device)
                basic = batch['basic'].to(self.device)
                count = batch['count'].to(self.device)
                sentiment = batch['sentiment'].to(self.device)
                momentum = batch['momentum'].to(self.device)
                decay = batch['decay'].to(self.device)
                control = batch['control'].to(self.device)
                dow = batch['dow'].to(self.device)
                month = batch['month'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(
                    sentiment_seq, basic, count, sentiment,
                    momentum, decay, control, dow, month
                )
                
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        predictions = np.concatenate(all_preds, axis=0).flatten()
        targets = np.concatenate(all_targets, axis=0).flatten()
        
        return predictions, targets
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model and return all metrics.
        """
        predictions, targets = self.predict(data_loader)
        
        metrics = MetricsCalculator.calculate_all(targets, predictions)
        
        return metrics


class StockReturnTrainer:
    """
    High-level trainer for stock return prediction.
    
    Handles the full pipeline for training models across different horizons.
    """
    
    def __init__(self,
                 output_dir: str = 'output',
                 lstm_hidden_size: int = 64,
                 lstm_num_layers: int = 1,
                 hidden_dims: Tuple[int, ...] = (128, 64, 32),
                 dropout: float = 0.3,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 batch_size: int = 32,
                 n_epochs: int = 100,
                 early_stopping_patience: int = 5,
                 device: Optional[torch.device] = None):
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.models: Dict[str, StockReturnPredictionModel] = {}
        self.results: Dict[str, Dict[str, float]] = {}
        self.histories: Dict[str, Dict[str, List[float]]] = {}
    
    def create_model(self, feature_dims: Optional[Dict[str, int]] = None) -> StockReturnPredictionModel:
        """Create a new model instance."""
        model = StockReturnPredictionModel(
            seq_len=7,
            lstm_hidden_size=self.lstm_hidden_size,
            lstm_num_layers=self.lstm_num_layers,
            lstm_dropout=self.dropout,
            feature_dims=feature_dims,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout
        )
        
        logger.info(f"\n{get_model_summary(model)}")
        
        return model
    
    def train_horizon(self,
                    train_data: Dict[str, np.ndarray],
                    val_data: Dict[str, np.ndarray],
                    horizon: str,
                    verbose: bool = True) -> Tuple[StockReturnPredictionModel, Dict[str, float]]:
        """
        Train model for a specific horizon.
        
        Args:
            train_data: Training features dictionary
            val_data: Validation features dictionary
            horizon: Horizon name (e.g., '1_DAY_RETURN')
        
        Returns:
            Tuple of (trained model, training history)
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Training horizon: {horizon}")
        logger.info(f"{'='*50}")
        
        feature_dims = {
            'basic': train_data['basic'].shape[1],
            'count': train_data['count'].shape[1],
            'sentiment': train_data['sentiment'].shape[1],
            'momentum': train_data['momentum'].shape[1],
            'decay': train_data['decay'].shape[1],
            'control': train_data['control'].shape[1],
            'dow': train_data['dow'].shape[1],
            'month': train_data['month'].shape[1]
        }
        
        model = self.create_model(feature_dims)
        
        train_dataset = StockReturnDataset(train_data)
        val_dataset = StockReturnDataset(val_data)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        trainer = Trainer(
            model=model,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            device=self.device
        )
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=self.n_epochs,
            early_stopping_patience=self.early_stopping_patience,
            verbose=verbose
        )
        
        return model, history
    
    def evaluate_horizon(self,
                       model: StockReturnPredictionModel,
                       test_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Evaluate model on test data.
        """
        test_dataset = StockReturnDataset(test_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
        )
        
        evaluator = Evaluator(model, device=self.device)
        metrics = evaluator.evaluate(test_loader)
        
        return metrics
    
    def train_all_horizons(self,
                          train_data: Dict[str, np.ndarray],
                          val_data: Dict[str, np.ndarray],
                          test_data: Dict[str, np.ndarray],
                          horizons: List[str] = None) -> pd.DataFrame:
        """
        Train and evaluate models for all horizons.
        
        Args:
            train_data: Training features
            val_data: Validation features
            test_data: Test features
            horizons: List of horizon columns to train
        
        Returns:
            DataFrame with evaluation results
        """
        if horizons is None:
            horizons = ['1_DAY_RETURN', '2_DAY_RETURN', '3_DAY_RETURN', '7_DAY_RETURN']
        
        results_list = []
        
        for horizon in horizons:
            target_col = horizon
            
            train_subset = {k: v for k, v in train_data.items()}
            val_subset = {k: v for k, v in val_data.items()}
            test_subset = {k: v for k, v in test_data.items()}
            
            train_subset['targets'] = train_data['targets']
            val_subset['targets'] = val_data['targets']
            test_subset['targets'] = test_data['targets']
            
            model, history = self.train_horizon(
                train_data=train_subset,
                val_data=val_subset,
                horizon=horizon
            )
            
            test_metrics = self.evaluate_horizon(model, test_subset)
            
            self.models[horizon] = model
            self.results[horizon] = test_metrics
            self.histories[horizon] = history
            
            result_row = {'Horizon': horizon}
            result_row.update(test_metrics)
            results_list.append(result_row)
            
            self._save_model(model, horizon)
        
        results_df = pd.DataFrame(results_list)
        results_df.to_csv(self.output_dir / 'evaluation_results.csv', index=False)
        
        logger.info("\n" + "="*60)
        logger.info("EVALUATION RESULTS")
        logger.info("="*60)
        self._print_results_table(results_df)
        
        self._save_results_summary()
        
        return results_df
    
    def _save_model(self, model: StockReturnPredictionModel, horizon: str):
        """Save model checkpoint."""
        save_path = self.output_dir / f'model_{horizon}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'horizon': horizon,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        logger.info(f"Model saved to {save_path}")
    
    def _print_results_table(self, results_df: pd.DataFrame):
        """Print formatted results table."""
        header = f"{'Horizon':<15} {'MSE':>10} {'RMSE':>10} {'MAE':>10} {'MAPE':>10} {'R2':>10} {'Dir_Acc':>10}"
        separator = "-" * len(header)
        
        print(f"\n{header}")
        print(separator)
        
        for _, row in results_df.iterrows():
            print(
                f"{row['Horizon']:<15} "
                f"{row['MSE']:>10.4f} "
                f"{row['RMSE']:>10.4f} "
                f"{row['MAE']:>10.4f} "
                f"{row['MAPE']:>10.2f} "
                f"{row['R2']:>10.4f} "
                f"{row['Directional_Accuracy']:>10.2f}"
            )
        
        print(separator)
    
    def _save_results_summary(self):
        """Save detailed results summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'config': {
                'lstm_hidden_size': self.lstm_hidden_size,
                'lstm_num_layers': self.lstm_num_layers,
                'hidden_dims': self.hidden_dims,
                'dropout': self.dropout,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                'batch_size': self.batch_size,
                'n_epochs': self.n_epochs,
                'early_stopping_patience': self.early_stopping_patience
            },
            'results': {k: {kk: float(vv) for kk, vv in v.items()} 
                       for k, v in self.results.items()}
        }
        
        with open(self.output_dir / 'results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def main():
    """Example usage of the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train stock return prediction models')
    parser.add_argument('--train', '-t', required=True, help='Training data (NumPy dict)')
    parser.add_argument('--val', '-v', required=True, help='Validation data (NumPy dict)')
    parser.add_argument('--test', required=True, help='Test data (NumPy dict)')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size')
    
    args = parser.parse_args()
    
    train_data = np.load(args.train, allow_pickle=True).item()
    val_data = np.load(args.val, allow_pickle=True).item()
    test_data = np.load(args.test, allow_pickle=True).item()
    
    trainer = StockReturnTrainer(
        output_dir=args.output,
        n_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    results = trainer.train_all_horizons(train_data, val_data, test_data)
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
