"""
Transfer Learning Model for Track Performance Prediction
Phase 2.2: Predict driver performance on new tracks based on known track performance

Architecture:
1. Source Track Encoder - Processes driver performance at known track
2. Track DNA Encoder - Processes target track characteristics
3. Transfer Network - Maps source performance → target track prediction
4. Performance Predictor - Outputs expected lap times, positions, etc.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SourceTrackEncoder(nn.Module):
    """
    Encodes driver performance at a known (source) track.
    Input: Driver skill vector + source track performance metrics
    Output: Encoded representation of driver-track interaction
    """
    
    def __init__(self, driver_embedding_dim: int = 8, performance_features: int = 5, hidden_dim: int = 64):
        """
        Initialize Source Track Encoder.
        
        Args:
            driver_embedding_dim: Dimension of driver skill vector (default: 8)
            performance_features: Number of performance features (position, lap time, etc.)
            hidden_dim: Hidden dimension size
        """
        super(SourceTrackEncoder, self).__init__()
        
        # Input: driver embedding + performance features
        input_dim = driver_embedding_dim + performance_features
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, driver_embedding: torch.Tensor, performance_features: torch.Tensor) -> torch.Tensor:
        """
        Encode source track performance.
        
        Args:
            driver_embedding: Driver skill vector [batch_size, driver_embedding_dim]
            performance_features: Performance metrics [batch_size, performance_features]
            
        Returns:
            Encoded representation [batch_size, hidden_dim // 2]
        """
        # Concatenate driver embedding and performance features
        x = torch.cat([driver_embedding, performance_features], dim=1)
        return self.encoder(x)


class TrackDNAEncoder(nn.Module):
    """
    Encodes target track DNA characteristics.
    Input: Track DNA features
    Output: Encoded track representation
    """
    
    def __init__(self, track_dna_dim: int = 20, hidden_dim: int = 64):
        """
        Initialize Track DNA Encoder.
        
        Args:
            track_dna_dim: Dimension of track DNA features
            hidden_dim: Hidden dimension size
        """
        super(TrackDNAEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(track_dna_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, track_dna: torch.Tensor) -> torch.Tensor:
        """
        Encode track DNA.
        
        Args:
            track_dna: Track DNA features [batch_size, track_dna_dim]
            
        Returns:
            Encoded track representation [batch_size, hidden_dim // 2]
        """
        return self.encoder(track_dna)


class TransferNetwork(nn.Module):
    """
    Transfer Network that maps source performance to target track prediction.
    Combines source track encoding and target track DNA encoding.
    """
    
    def __init__(self, source_dim: int = 32, target_dim: int = 32, hidden_dim: int = 128):
        """
        Initialize Transfer Network.
        
        Args:
            source_dim: Dimension of source track encoding
            target_dim: Dimension of target track DNA encoding
            hidden_dim: Hidden dimension size
        """
        super(TransferNetwork, self).__init__()
        
        # Combine source and target encodings
        combined_dim = source_dim + target_dim
        
        self.transfer = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, source_encoding: torch.Tensor, target_encoding: torch.Tensor) -> torch.Tensor:
        """
        Transfer source performance to target track.
        
        Args:
            source_encoding: Source track encoding [batch_size, source_dim]
            target_encoding: Target track DNA encoding [batch_size, target_dim]
            
        Returns:
            Transfer representation [batch_size, hidden_dim // 2]
        """
        # Concatenate source and target
        combined = torch.cat([source_encoding, target_encoding], dim=1)
        return self.transfer(combined)


class PerformancePredictor(nn.Module):
    """
    Predicts performance metrics on target track.
    Outputs: lap times, positions, speeds, etc.
    """
    
    def __init__(self, input_dim: int = 64, output_dim: int = 4):
        """
        Initialize Performance Predictor.
        
        Args:
            input_dim: Dimension of transfer representation
            output_dim: Number of output predictions (lap_time, position, speed, finish_probability)
        """
        super(PerformancePredictor, self).__init__()
        
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, transfer_representation: torch.Tensor) -> torch.Tensor:
        """
        Predict performance on target track.
        
        Args:
            transfer_representation: Transfer network output [batch_size, input_dim]
            
        Returns:
            Performance predictions [batch_size, output_dim]
            [predicted_lap_time, predicted_position, predicted_speed, finish_probability]
        """
        return self.predictor(transfer_representation)


class TrackTransferModel(nn.Module):
    """
    Complete Transfer Learning Model for track performance prediction.
    Combines all components: Source Encoder, DNA Encoder, Transfer Network, Predictor
    """
    
    def __init__(
        self,
        driver_embedding_dim: int = 8,
        track_dna_dim: int = 20,
        performance_features: int = 5,
        hidden_dim: int = 64
    ):
        """
        Initialize complete transfer learning model.
        
        Args:
            driver_embedding_dim: Dimension of driver skill vector
            track_dna_dim: Dimension of track DNA features
            performance_features: Number of performance features from source track
            hidden_dim: Hidden dimension size
        """
        super(TrackTransferModel, self).__init__()
        
        self.source_encoder = SourceTrackEncoder(
            driver_embedding_dim=driver_embedding_dim,
            performance_features=performance_features,
            hidden_dim=hidden_dim
        )
        
        self.dna_encoder = TrackDNAEncoder(
            track_dna_dim=track_dna_dim,
            hidden_dim=hidden_dim
        )
        
        self.transfer_network = TransferNetwork(
            source_dim=hidden_dim // 2,
            target_dim=hidden_dim // 2,
            hidden_dim=hidden_dim * 2
        )
        
        self.predictor = PerformancePredictor(
            input_dim=hidden_dim,
            output_dim=4  # lap_time, position, speed, finish_probability
        )
    
    def forward(
        self,
        driver_embedding: torch.Tensor,
        source_performance: torch.Tensor,
        target_track_dna: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: predict performance on target track.
        
        Args:
            driver_embedding: Driver skill vector [batch_size, driver_embedding_dim]
            source_performance: Performance at source track [batch_size, performance_features]
            target_track_dna: Target track DNA features [batch_size, track_dna_dim]
            
        Returns:
            Performance predictions [batch_size, 4]
            [predicted_lap_time, predicted_position, predicted_speed, finish_probability]
        """
        # Encode source track performance
        source_encoding = self.source_encoder(driver_embedding, source_performance)
        
        # Encode target track DNA
        target_encoding = self.dna_encoder(target_track_dna)
        
        # Transfer from source to target
        transfer_rep = self.transfer_network(source_encoding, target_encoding)
        
        # Predict performance
        predictions = self.predictor(transfer_rep)
        
        return predictions
    
    def predict(
        self,
        driver_embedding: np.ndarray,
        source_performance: np.ndarray,
        target_track_dna: np.ndarray
    ) -> np.ndarray:
        """
        Predict performance (numpy interface).
        
        Args:
            driver_embedding: Driver skill vector
            source_performance: Performance at source track
            target_track_dna: Target track DNA features
            
        Returns:
            Performance predictions as numpy array
        """
        self.eval()
        with torch.no_grad():
            # Convert to tensors
            driver_tensor = torch.FloatTensor(driver_embedding).unsqueeze(0)
            source_tensor = torch.FloatTensor(source_performance).unsqueeze(0)
            target_tensor = torch.FloatTensor(target_track_dna).unsqueeze(0)
            
            # Predict
            predictions = self.forward(driver_tensor, source_tensor, target_tensor)
            
            return predictions.squeeze(0).numpy()


class TransferLearningTrainer:
    """
    Trainer for the transfer learning model.
    Handles data preparation, training, and validation.
    """
    
    def __init__(
        self,
        model: TrackTransferModel,
        learning_rate: float = 0.001,
        device: Optional[torch.device] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: TrackTransferModel instance
            learning_rate: Learning rate for optimizer
            device: Device to run on (None = auto-detect)
        """
        self.model = model
        self.device = device or (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model.to(self.device)
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.criterion = nn.MSELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def prepare_training_data(
        self,
        driver_embeddings: pd.DataFrame,
        track_dna_data: pd.DataFrame,
        performance_data: pd.DataFrame
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare training data from driver embeddings, track DNA, and performance data.
        
        Args:
            driver_embeddings: DataFrame with driver embeddings
            track_dna_data: DataFrame with track DNA features
            performance_data: DataFrame with actual performance (for labels)
            
        Returns:
            Tuple of (driver_embeddings, source_performance, target_dna, labels)
        """
        # This is a placeholder - actual implementation would process the data
        # For now, return empty tensors as structure
        return (
            torch.FloatTensor([]),
            torch.FloatTensor([]),
            torch.FloatTensor([]),
            torch.FloatTensor([])
        )
    
    def train_epoch(
        self,
        driver_embeddings: torch.Tensor,
        source_performance: torch.Tensor,
        target_dna: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Train for one epoch.
        
        Args:
            driver_embeddings: Driver skill vectors
            source_performance: Source track performance
            target_dna: Target track DNA
            labels: Target performance (ground truth)
            batch_size: Batch size
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Move to device
        driver_embeddings = driver_embeddings.to(self.device)
        source_performance = source_performance.to(self.device)
        target_dna = target_dna.to(self.device)
        labels = labels.to(self.device)
        
        # Batch training
        dataset_size = driver_embeddings.size(0)
        for i in range(0, dataset_size, batch_size):
            end_idx = min(i + batch_size, dataset_size)
            
            batch_driver = driver_embeddings[i:end_idx]
            batch_source = source_performance[i:end_idx]
            batch_target = target_dna[i:end_idx]
            batch_labels = labels[i:end_idx]
            
            # Forward pass
            predictions = self.model(batch_driver, batch_source, batch_target)
            
            # Calculate loss
            loss = self.criterion(predictions, batch_labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def validate(
        self,
        driver_embeddings: torch.Tensor,
        source_performance: torch.Tensor,
        target_dna: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Validate model.
        
        Args:
            driver_embeddings: Driver skill vectors
            source_performance: Source track performance
            target_dna: Target track DNA
            labels: Target performance (ground truth)
            batch_size: Batch size
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        # Move to device
        driver_embeddings = driver_embeddings.to(self.device)
        source_performance = source_performance.to(self.device)
        target_dna = target_dna.to(self.device)
        labels = labels.to(self.device)
        
        with torch.no_grad():
            dataset_size = driver_embeddings.size(0)
            for i in range(0, dataset_size, batch_size):
                end_idx = min(i + batch_size, dataset_size)
                
                batch_driver = driver_embeddings[i:end_idx]
                batch_source = source_performance[i:end_idx]
                batch_target = target_dna[i:end_idx]
                batch_labels = labels[i:end_idx]
                
                # Forward pass
                predictions = self.model(batch_driver, batch_source, batch_target)
                
                # Calculate loss
                loss = self.criterion(predictions, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def save_model(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from file."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Model loaded from {path}")


def create_transfer_model(
    driver_embedding_dim: int = 8,
    track_dna_dim: int = 20,
    performance_features: int = 5
) -> TrackTransferModel:
    """
    Convenience function to create a transfer learning model.
    
    Args:
        driver_embedding_dim: Dimension of driver skill vector
        track_dna_dim: Dimension of track DNA features
        performance_features: Number of performance features
        
    Returns:
        Initialized TrackTransferModel
    """
    return TrackTransferModel(
        driver_embedding_dim=driver_embedding_dim,
        track_dna_dim=track_dna_dim,
        performance_features=performance_features
    )

