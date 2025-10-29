"""
Federated Multi-Task Learning (FMTL) Model for FedRoute Framework

This module implements the unified FMTL architecture that learns both path
and music recommendations simultaneously in a federated learning setting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class ContextEncoder(nn.Module):
    """
    Shared context encoder that processes vehicle and environmental data.
    This is the core component that enables knowledge transfer between tasks.
    """
    
    def __init__(self, 
                 input_dim: int = 64,
                 hidden_dims: List[int] = [128, 256, 128],
                 dropout_rate: float = 0.2):
        super(ContextEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout_rate
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
            
        self.encoder = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the context encoder.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Contains: [lat, lon, speed, time_of_day, day_of_week, 
                         weather, road_type, traffic_density]
        
        Returns:
            Encoded context representation of shape (batch_size, output_dim)
        """
        return self.encoder(x)


class PathRecommendationHead(nn.Module):
    """
    Task-specific head for path and POI recommendations.
    """
    
    def __init__(self, 
                 context_dim: int = 128,
                 hidden_dims: List[int] = [64, 32],
                 num_poi_categories: int = 50,
                 num_pois: int = 1000):
        super(PathRecommendationHead, self).__init__()
        
        self.context_dim = context_dim
        self.num_poi_categories = num_poi_categories
        self.num_pois = num_pois
        
        # Build path recommendation layers
        layers = []
        prev_dim = context_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        self.path_layers = nn.Sequential(*layers)
        
        # Output layers for different prediction tasks
        self.poi_category_classifier = nn.Linear(prev_dim, num_poi_categories)
        self.poi_ranker = nn.Linear(prev_dim, num_pois)
        
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for path recommendations.
        
        Args:
            context: Encoded context from shared encoder
            
        Returns:
            Dictionary containing:
            - poi_categories: Logits for POI category prediction
            - poi_ranking: Logits for specific POI ranking
        """
        path_features = self.path_layers(context)
        
        return {
            'poi_categories': self.poi_category_classifier(path_features),
            'poi_ranking': self.poi_ranker(path_features)
        }


class MusicRecommendationHead(nn.Module):
    """
    Task-specific head for contextual music recommendations.
    """
    
    def __init__(self, 
                 context_dim: int = 128,
                 hidden_dims: List[int] = [64, 32],
                 num_genres: int = 20,
                 num_artists: int = 500,
                 num_tracks: int = 10000):
        super(MusicRecommendationHead, self).__init__()
        
        self.context_dim = context_dim
        self.num_genres = num_genres
        self.num_artists = num_artists
        self.num_tracks = num_tracks
        
        # Build music recommendation layers
        layers = []
        prev_dim = context_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        self.music_layers = nn.Sequential(*layers)
        
        # Output layers for different music prediction tasks
        self.genre_classifier = nn.Linear(prev_dim, num_genres)
        self.artist_ranker = nn.Linear(prev_dim, num_artists)
        self.track_ranker = nn.Linear(prev_dim, num_tracks)
        
    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for music recommendations.
        
        Args:
            context: Encoded context from shared encoder
            
        Returns:
            Dictionary containing:
            - genres: Logits for genre prediction
            - artists: Logits for artist ranking
            - tracks: Logits for track ranking
        """
        music_features = self.music_layers(context)
        
        return {
            'genres': self.genre_classifier(music_features),
            'artists': self.artist_ranker(music_features),
            'tracks': self.track_ranker(music_features)
        }


class FedRouteFMTL(nn.Module):
    """
    Main FMTL model that combines context encoding with task-specific heads.
    This is the core innovation of the FedRoute framework.
    """
    
    def __init__(self, 
                 context_input_dim: int = 64,
                 context_hidden_dims: List[int] = [128, 256, 128],
                 path_hidden_dims: List[int] = [64, 32],
                 music_hidden_dims: List[int] = [64, 32],
                 num_poi_categories: int = 50,
                 num_pois: int = 1000,
                 num_genres: int = 20,
                 num_artists: int = 500,
                 num_tracks: int = 10000,
                 dropout_rate: float = 0.2):
        super(FedRouteFMTL, self).__init__()
        
        # Shared context encoder
        self.context_encoder = ContextEncoder(
            input_dim=context_input_dim,
            hidden_dims=context_hidden_dims,
            dropout_rate=dropout_rate
        )
        
        # Task-specific heads
        self.path_head = PathRecommendationHead(
            context_dim=context_hidden_dims[-1],
            hidden_dims=path_hidden_dims,
            num_poi_categories=num_poi_categories,
            num_pois=num_pois
        )
        
        self.music_head = MusicRecommendationHead(
            context_dim=context_hidden_dims[-1],
            hidden_dims=music_hidden_dims,
            num_genres=num_genres,
            num_artists=num_artists,
            num_tracks=num_tracks
        )
        
        # Task weights for joint loss
        self.task_weights = nn.Parameter(torch.tensor([1.0, 1.0]))  # [path, music]
        
    def forward(self, context: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Forward pass through the complete FMTL model.
        
        Args:
            context: Input context tensor of shape (batch_size, context_input_dim)
            
        Returns:
            Dictionary containing outputs from both task heads
        """
        # Encode shared context
        encoded_context = self.context_encoder(context)
        
        # Get task-specific predictions
        path_outputs = self.path_head(encoded_context)
        music_outputs = self.music_head(encoded_context)
        
        return {
            'path': path_outputs,
            'music': music_outputs
        }
    
    def compute_joint_loss(self, 
                          outputs: Dict[str, Dict[str, torch.Tensor]], 
                          targets: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Compute the joint loss for both tasks with learnable weights.
        
        Args:
            outputs: Model predictions from forward pass
            targets: Ground truth labels
            
        Returns:
            Weighted joint loss
        """
        # Path recommendation losses
        path_losses = {}
        if 'poi_categories' in targets['path']:
            path_losses['poi_cat'] = F.cross_entropy(
                outputs['path']['poi_categories'], 
                targets['path']['poi_categories']
            )
        if 'poi_ranking' in targets['path']:
            path_losses['poi_rank'] = F.binary_cross_entropy_with_logits(
                outputs['path']['poi_ranking'], 
                targets['path']['poi_ranking'].float()
            )
        
        # Music recommendation losses
        music_losses = {}
        if 'genres' in targets['music']:
            music_losses['genre'] = F.cross_entropy(
                outputs['music']['genres'], 
                targets['music']['genres']
            )
        if 'artists' in targets['music']:
            music_losses['artist'] = F.binary_cross_entropy_with_logits(
                outputs['music']['artists'], 
                targets['music']['artists'].float()
            )
        if 'tracks' in targets['music']:
            music_losses['track'] = F.binary_cross_entropy_with_logits(
                outputs['music']['tracks'], 
                targets['music']['tracks'].float()
            )
        
        # Compute weighted joint loss
        path_total_loss = sum(path_losses.values())
        music_total_loss = sum(music_losses.values())
        
        joint_loss = (self.task_weights[0] * path_total_loss + 
                     self.task_weights[1] * music_total_loss)
        
        return joint_loss, {
            'path_loss': path_total_loss,
            'music_loss': music_total_loss,
            'path_components': path_losses,
            'music_components': music_losses
        }
    
    def get_shared_parameters(self) -> List[torch.Tensor]:
        """Get parameters from the shared context encoder."""
        return list(self.context_encoder.parameters())
    
    def get_task_specific_parameters(self, task: str) -> List[torch.Tensor]:
        """Get parameters from task-specific heads."""
        if task == 'path':
            return list(self.path_head.parameters())
        elif task == 'music':
            return list(self.music_head.parameters())
        else:
            raise ValueError(f"Unknown task: {task}")


class AttentionMechanism(nn.Module):
    """
    Attention mechanism for cross-task knowledge transfer.
    Allows one task to attend to features learned by the other task.
    """
    
    def __init__(self, context_dim: int = 128, attention_dim: int = 64):
        super(AttentionMechanism, self).__init__()
        
        self.context_dim = context_dim
        self.attention_dim = attention_dim
        
        # Attention layers
        self.query_projection = nn.Linear(context_dim, attention_dim)
        self.key_projection = nn.Linear(context_dim, attention_dim)
        self.value_projection = nn.Linear(context_dim, attention_dim)
        self.output_projection = nn.Linear(attention_dim, context_dim)
        
    def forward(self, path_context: torch.Tensor, music_context: torch.Tensor) -> torch.Tensor:
        """
        Compute attention between path and music contexts.
        
        Args:
            path_context: Context from path recommendation head
            music_context: Context from music recommendation head
            
        Returns:
            Attended context representation
        """
        # Project to attention space
        Q = self.query_projection(path_context)
        K = self.key_projection(music_context)
        V = self.value_projection(music_context)
        
        # Compute attention weights
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.attention_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        attended_values = torch.matmul(attention_weights, V)
        output = self.output_projection(attended_values)
        
        return output


def create_fedroute_model(config: Dict) -> FedRouteFMTL:
    """
    Factory function to create a FedRoute FMTL model from configuration.
    
    Args:
        config: Configuration dictionary containing model parameters
        
    Returns:
        Initialized FedRouteFMTL model
    """
    return FedRouteFMTL(
        context_input_dim=config.get('context_input_dim', 64),
        context_hidden_dims=config.get('context_hidden_dims', [128, 256, 128]),
        path_hidden_dims=config.get('path_hidden_dims', [64, 32]),
        music_hidden_dims=config.get('music_hidden_dims', [64, 32]),
        num_poi_categories=config.get('num_poi_categories', 50),
        num_pois=config.get('num_pois', 1000),
        num_genres=config.get('num_genres', 20),
        num_artists=config.get('num_artists', 500),
        num_tracks=config.get('num_tracks', 10000),
        dropout_rate=config.get('dropout_rate', 0.2)
    )


if __name__ == "__main__":
    # Example usage and testing
    config = {
        'context_input_dim': 64,
        'context_hidden_dims': [128, 256, 128],
        'path_hidden_dims': [64, 32],
        'music_hidden_dims': [64, 32],
        'num_poi_categories': 50,
        'num_pois': 1000,
        'num_genres': 20,
        'num_artists': 500,
        'num_tracks': 10000
    }
    
    model = create_fedroute_model(config)
    
    # Test forward pass
    batch_size = 32
    context_input = torch.randn(batch_size, 64)
    
    outputs = model(context_input)
    print("Model outputs structure:")
    for task, task_outputs in outputs.items():
        print(f"  {task}:")
        for output_name, output_tensor in task_outputs.items():
            print(f"    {output_name}: {output_tensor.shape}")
    
    # Test loss computation
    targets = {
        'path': {
            'poi_categories': torch.randint(0, 50, (batch_size,)),
            'poi_ranking': torch.randint(0, 2, (batch_size, 1000)).float()
        },
        'music': {
            'genres': torch.randint(0, 20, (batch_size,)),
            'artists': torch.randint(0, 2, (batch_size, 500)).float(),
            'tracks': torch.randint(0, 2, (batch_size, 10000)).float()
        }
    }
    
    joint_loss, loss_components = model.compute_joint_loss(outputs, targets)
    print(f"\nJoint loss: {joint_loss.item():.4f}")
    print(f"Path loss: {loss_components['path_loss'].item():.4f}")
    print(f"Music loss: {loss_components['music_loss'].item():.4f}")

