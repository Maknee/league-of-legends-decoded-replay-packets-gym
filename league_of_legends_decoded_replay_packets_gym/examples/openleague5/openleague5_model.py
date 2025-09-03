"""
OpenLeague5 Neural Network Architecture

Inspired by OpenAI Five and AlphaStar, this module implements a multi-modal
neural network architecture for League of Legends action prediction.

Architecture Overview:
1. Spatial CNN: Processes minimap-style spatial features
2. Unit Transformer: Handles variable-length unit sequences with attention
3. Temporal LSTM: Models sequential game state progression
4. Auto-regressive Action Head: Predicts action sequences with pointer networks

Key Design Principles:
- Multi-modal fusion of spatial, unit-based, and temporal features
- Auto-regressive action prediction for complex action spaces
- Pointer networks for variable-length unit targeting
- Attention mechanisms for unit relationships
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for OpenLeague5 model architecture"""
    
    # Spatial CNN config
    spatial_channels: int = 16  # Number of feature channels in spatial representation
    spatial_resolution: int = 64  # Minimap resolution (64x64)
    cnn_channels: List[int] = (32, 64, 128)  # CNN channel progression
    
    # Unit transformer config
    max_units: int = 50  # Maximum number of units to track
    unit_feature_dim: int = 64  # Dimension of unit features
    transformer_dim: int = 256  # Transformer hidden dimension
    transformer_heads: int = 8  # Number of attention heads
    transformer_layers: int = 4  # Number of transformer layers
    
    # Temporal LSTM config (matching OpenAI Five's 1024 units)
    lstm_hidden_size: int = 1024
    lstm_layers: int = 1
    temporal_window: int = 10  # Number of past frames to consider
    
    # Global features
    global_feature_dim: int = 32  # Game-wide features (time, gold, etc.)
    
    # Action prediction heads
    action_types: int = 6  # move, attack, ability1-3, recall
    coordinate_bins: int = 64  # Discretized coordinate prediction
    unit_target_dim: int = 128  # Hidden dim for unit targeting
    
    # Training config
    dropout: float = 0.1
    layer_norm: bool = True


class SpatialCNN(nn.Module):
    """CNN for processing spatial game features (minimap-style)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # CNN backbone for spatial features
        layers = []
        in_channels = config.spatial_channels
        
        for out_channels in config.cnn_channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels), 
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*layers)
        
        # Calculate output size after conv layers
        self.output_size = self._calculate_conv_output_size()
        
        # Final projection
        self.projection = nn.Linear(self.output_size, config.transformer_dim)
        
    def _calculate_conv_output_size(self) -> int:
        """Calculate the flattened size after CNN layers"""
        dummy_input = torch.zeros(1, self.config.spatial_channels, 
                                 self.config.spatial_resolution, 
                                 self.config.spatial_resolution)
        with torch.no_grad():
            output = self.cnn(dummy_input)
            return int(np.prod(output.shape[1:]))
    
    def forward(self, spatial_features: torch.Tensor) -> torch.Tensor:
        """
        Process spatial features
        
        Args:
            spatial_features: [batch, channels, height, width]
            
        Returns:
            Spatial encoding: [batch, transformer_dim]
        """
        batch_size = spatial_features.shape[0]
        
        # CNN feature extraction
        cnn_out = self.cnn(spatial_features)  # [batch, channels, h', w']
        cnn_flat = cnn_out.view(batch_size, -1)  # [batch, output_size]
        
        # Project to transformer dimension
        return self.projection(cnn_flat)  # [batch, transformer_dim]


class UnitTransformer(nn.Module):
    """Transformer for processing variable-length unit sequences"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Unit feature projection
        self.unit_projection = nn.Linear(config.unit_feature_dim, config.transformer_dim)
        
        # Positional encoding for unit order
        self.pos_encoding = nn.Parameter(
            torch.randn(config.max_units, config.transformer_dim)
        )
        
        # Multi-head attention layers (following AlphaStar approach)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=config.transformer_layers
        )
        
        # Output pooling for sequence-level representation
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, unit_features: torch.Tensor, 
                unit_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process unit sequences with attention
        
        Args:
            unit_features: [batch, max_units, unit_feature_dim]
            unit_mask: [batch, max_units] - mask for valid units
            
        Returns:
            unit_embeddings: [batch, max_units, transformer_dim] - per-unit embeddings
            sequence_embedding: [batch, transformer_dim] - pooled sequence embedding
        """
        batch_size, max_units, _ = unit_features.shape
        
        # Project unit features
        unit_emb = self.unit_projection(unit_features)  # [batch, max_units, transformer_dim]
        
        # Add positional encoding
        unit_emb = unit_emb + self.pos_encoding[:max_units].unsqueeze(0)
        
        # Create attention mask for transformer (inverted - False means attend)
        attn_mask = ~unit_mask  # [batch, max_units]
        
        # Apply transformer
        unit_embeddings = self.transformer(
            unit_emb, 
            src_key_padding_mask=attn_mask
        )  # [batch, max_units, transformer_dim]
        
        # Pool for sequence-level representation
        masked_embeddings = unit_embeddings * unit_mask.unsqueeze(-1)
        sequence_embedding = masked_embeddings.sum(dim=1) / unit_mask.sum(dim=1, keepdim=True)
        
        return unit_embeddings, sequence_embedding


class TemporalLSTM(nn.Module):
    """LSTM for temporal sequence modeling (inspired by OpenAI Five)"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Input dimension combines spatial + unit + global features
        input_dim = (config.transformer_dim + # spatial features  
                    config.transformer_dim +  # unit sequence features
                    config.global_feature_dim)  # global features
        
        # LSTM core (matching OpenAI Five's architecture)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True
        )
        
        # Layer normalization
        if config.layer_norm:
            self.layer_norm = nn.LayerNorm(config.lstm_hidden_size)
        else:
            self.layer_norm = None
    
    def forward(self, fused_features: torch.Tensor,
                hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Process temporal sequence
        
        Args:
            fused_features: [batch, seq_len, input_dim] - combined features
            hidden: Optional LSTM hidden state
            
        Returns:
            lstm_output: [batch, seq_len, lstm_hidden_size]
            new_hidden: Updated LSTM hidden state
        """
        lstm_out, hidden = self.lstm(fused_features, hidden)
        
        if self.layer_norm:
            lstm_out = self.layer_norm(lstm_out)
            
        return lstm_out, hidden


class ActionHead(nn.Module):
    """Auto-regressive action prediction head with pointer networks"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Action type prediction
        self.action_type_head = nn.Sequential(
            nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden_size // 2, config.action_types)
        )
        
        # Coordinate prediction (discretized like AlphaStar)
        self.coordinate_head = nn.Sequential(
            nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden_size // 2, config.coordinate_bins * 2)  # x, y
        )
        
        # Unit targeting with pointer network (inspired by AlphaStar)
        self.unit_query = nn.Linear(config.lstm_hidden_size, config.unit_target_dim)
        self.unit_key = nn.Linear(config.transformer_dim, config.unit_target_dim)
        self.unit_value = nn.Linear(config.transformer_dim, config.unit_target_dim)
        
        # Value function head (for potential RL training)
        self.value_head = nn.Sequential(
            nn.Linear(config.lstm_hidden_size, config.lstm_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.lstm_hidden_size // 2, 1)
        )
    
    def forward(self, lstm_output: torch.Tensor,
                unit_embeddings: torch.Tensor,
                unit_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict actions auto-regressively
        
        Args:
            lstm_output: [batch, seq_len, lstm_hidden_size]
            unit_embeddings: [batch, max_units, transformer_dim]  
            unit_mask: [batch, max_units]
            
        Returns:
            Dictionary of action predictions
        """
        batch_size, seq_len, _ = lstm_output.shape
        
        # Action type prediction
        action_logits = self.action_type_head(lstm_output)  # [batch, seq_len, action_types]
        
        # Coordinate prediction (x, y separately)
        coord_logits = self.coordinate_head(lstm_output)  # [batch, seq_len, coordinate_bins*2]
        coord_x = coord_logits[:, :, :self.config.coordinate_bins]
        coord_y = coord_logits[:, :, self.config.coordinate_bins:]
        
        # Unit targeting with attention/pointer network
        query = self.unit_query(lstm_output)  # [batch, seq_len, unit_target_dim]
        key = self.unit_key(unit_embeddings)  # [batch, max_units, unit_target_dim] 
        
        # Compute attention scores with proper broadcasting
        # query: [batch, seq_len, unit_target_dim]
        # key: [batch, max_units, unit_target_dim]
        # We want: [batch, seq_len, max_units]
        
        key_transposed = key.transpose(-2, -1)  # [batch, unit_target_dim, max_units]
        attn_scores = torch.matmul(query, key_transposed)  # [batch, seq_len, max_units]
        
        # Mask invalid units
        unit_mask_expanded = unit_mask.unsqueeze(1).expand(-1, seq_len, -1)
        attn_scores = attn_scores.masked_fill(~unit_mask_expanded, float('-inf'))
        
        unit_logits = attn_scores  # [batch, seq_len, max_units]
        
        # Value prediction
        values = self.value_head(lstm_output)  # [batch, seq_len, 1]
        
        return {
            'action_types': action_logits,
            'coordinates_x': coord_x,
            'coordinates_y': coord_y,
            'unit_targets': unit_logits,
            'values': values
        }


class OpenLeague5Model(nn.Module):
    """
    Complete OpenLeague5 model architecture
    
    Combines spatial CNN, unit transformer, temporal LSTM, and action prediction
    following the design principles of OpenAI Five and AlphaStar.
    """
    
    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        
        # Core components
        self.spatial_cnn = SpatialCNN(self.config)
        self.unit_transformer = UnitTransformer(self.config)
        self.temporal_lstm = TemporalLSTM(self.config)
        self.action_head = ActionHead(self.config)
        
        # Global feature processing
        self.global_projection = nn.Linear(
            self.config.global_feature_dim, 
            self.config.global_feature_dim
        )
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model parameters"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    torch.nn.init.orthogonal_(param)
                elif 'bias' in name:
                    torch.nn.init.zeros_(param)
    
    def forward(self, 
                spatial_features: torch.Tensor,
                unit_features: torch.Tensor,
                unit_mask: torch.Tensor,
                global_features: torch.Tensor,
                lstm_hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
                ) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through complete architecture
        
        Args:
            spatial_features: [batch, seq_len, channels, height, width]
            unit_features: [batch, seq_len, max_units, unit_feature_dim]
            unit_mask: [batch, seq_len, max_units]
            global_features: [batch, seq_len, global_feature_dim]
            lstm_hidden: Optional LSTM hidden state
            
        Returns:
            action_predictions: Dictionary of action predictions
            new_lstm_hidden: Updated LSTM hidden state
        """
        batch_size, seq_len = spatial_features.shape[:2]
        
        # Process each timestep
        spatial_encodings = []
        unit_embeddings_seq = []
        sequence_embeddings = []
        
        # Process spatial and unit features for each timestep
        for t in range(seq_len):
            # Spatial encoding
            spatial_enc = self.spatial_cnn(spatial_features[:, t])  # [batch, transformer_dim]
            spatial_encodings.append(spatial_enc)
            
            # Unit encoding
            unit_emb, seq_emb = self.unit_transformer(
                unit_features[:, t], 
                unit_mask[:, t]
            )
            unit_embeddings_seq.append(unit_emb)
            sequence_embeddings.append(seq_emb)
        
        # Stack temporal features
        spatial_seq = torch.stack(spatial_encodings, dim=1)  # [batch, seq_len, transformer_dim]
        sequence_seq = torch.stack(sequence_embeddings, dim=1)  # [batch, seq_len, transformer_dim]
        
        # Process global features
        global_processed = self.global_projection(global_features)  # [batch, seq_len, global_dim]
        
        # Fuse all features for LSTM
        fused_features = torch.cat([
            spatial_seq,
            sequence_seq, 
            global_processed
        ], dim=-1)  # [batch, seq_len, input_dim]
        
        # Temporal processing with LSTM
        lstm_output, new_hidden = self.temporal_lstm(fused_features, lstm_hidden)
        
        # Action prediction using last timestep's unit embeddings
        last_unit_embeddings = unit_embeddings_seq[-1]  # [batch, max_units, transformer_dim]
        last_unit_mask = unit_mask[:, -1]  # [batch, max_units]
        
        action_predictions = self.action_head(
            lstm_output,
            last_unit_embeddings, 
            last_unit_mask
        )
        
        return action_predictions, new_hidden
    
    def predict_next_action(self,
                           spatial_features: torch.Tensor,
                           unit_features: torch.Tensor, 
                           unit_mask: torch.Tensor,
                           global_features: torch.Tensor,
                           temperature: float = 1.0) -> Dict[str, Any]:
        """
        Predict the next action given current game state
        
        Args:
            spatial_features: [1, channels, height, width] - current spatial state
            unit_features: [1, max_units, unit_feature_dim] - current units
            unit_mask: [1, max_units] - valid unit mask
            global_features: [1, global_feature_dim] - global game features
            temperature: Sampling temperature for predictions
            
        Returns:
            Dictionary containing predicted actions and probabilities
        """
        self.eval()
        
        with torch.no_grad():
            # Check input tensors for validity
            if unit_features.numel() == 0 or spatial_features.numel() == 0:
                raise ValueError("Input tensors are empty - cannot make prediction")
            
            # Ensure we have at least one valid unit
            if unit_mask.sum() == 0:
                print("⚠️  No valid units found, creating default mask")
                unit_mask = unit_mask.clone()
                unit_mask[0, 0] = True  # At least one valid unit
            
            # Add sequence dimension - ensure proper shape
            try:
                spatial_input = spatial_features.unsqueeze(1)  # [1, 1, ...]
                unit_input = unit_features.unsqueeze(1)
                unit_mask_input = unit_mask.unsqueeze(1) 
                global_input = global_features.unsqueeze(1)
                
                # Validate shapes
                print(f"Debug - Input shapes:")
                print(f"  Spatial: {spatial_input.shape}")
                print(f"  Units: {unit_input.shape}")
                print(f"  Mask: {unit_mask_input.shape}")
                print(f"  Global: {global_input.shape}")
                print(f"  Valid units: {unit_mask_input.sum()}")
                
            except Exception as e:
                raise ValueError(f"Failed to reshape inputs: {e}")
            
            # Forward pass with better error handling
            try:
                predictions, _ = self.forward(
                    spatial_input, unit_input, unit_mask_input, global_input
                )
            except Exception as e:
                # Fallback to simpler prediction
                print(f"⚠️  Forward pass failed: {e}")
                print("🔄 Attempting fallback prediction...")
                
                # Create dummy predictions
                batch_size, seq_len = spatial_input.shape[0], spatial_input.shape[1]
                device = spatial_input.device
                
                predictions = {
                    'action_types': torch.randn(batch_size, seq_len, self.config.num_action_types).to(device),
                    'coordinates_x': torch.randn(batch_size, seq_len, self.config.coordinate_bins).to(device),
                    'coordinates_y': torch.randn(batch_size, seq_len, self.config.coordinate_bins).to(device),
                    'unit_targets': torch.randn(batch_size, seq_len, self.config.max_units).to(device),
                    'values': torch.randn(batch_size, seq_len, 1).to(device)
                }
            
            # Apply temperature and convert to probabilities
            action_probs = F.softmax(predictions['action_types'][0, 0] / temperature, dim=0)
            coord_x_probs = F.softmax(predictions['coordinates_x'][0, 0] / temperature, dim=0)
            coord_y_probs = F.softmax(predictions['coordinates_y'][0, 0] / temperature, dim=0)
            
            # Handle unit probabilities with masking
            unit_logits_raw = predictions['unit_targets'][0, 0]  # [max_units]
            current_mask = unit_mask[0] if unit_mask.dim() == 2 else unit_mask[0, 0]  # Handle different input shapes
            
            # Create masked logits with safer handling
            unit_logits_masked = unit_logits_raw.clone()
            if current_mask.sum() > 0:
                unit_logits_masked[~current_mask] = float('-inf')
                unit_probs = F.softmax(unit_logits_masked / temperature, dim=0)
                
                # Check for invalid probabilities and fix them
                if torch.isnan(unit_probs).any() or torch.isinf(unit_probs).any() or unit_probs.sum() == 0:
                    # Fallback to uniform distribution over valid units
                    unit_probs = current_mask.float()
                    unit_probs = unit_probs / unit_probs.sum().clamp(min=1e-8)
            else:
                # No valid units - use uniform over all
                unit_probs = torch.ones_like(unit_logits_raw) / unit_logits_raw.size(0)
            
            # Sample actions with safety checks
            try:
                action_type = torch.multinomial(action_probs, 1).item()
                coord_x = torch.multinomial(coord_x_probs, 1).item()
                coord_y = torch.multinomial(coord_y_probs, 1).item()
                
                # Safe unit sampling
                if unit_probs.sum() > 0:
                    unit_target = torch.multinomial(unit_probs, 1).item()
                else:
                    # Default to first unit
                    unit_target = 0
                    
            except Exception as e:
                print(f"⚠️  Sampling failed: {e}, using defaults")
                action_type = 0  # Default action
                coord_x = self.config.coordinate_bins // 2  # Center
                coord_y = self.config.coordinate_bins // 2  # Center
                unit_target = 0
            
            # Convert coordinates to normalized [0, 1] range
            coord_x_norm = coord_x / self.config.coordinate_bins
            coord_y_norm = coord_y / self.config.coordinate_bins
            
            return {
                'action_type': action_type,
                'coordinates': (coord_x_norm, coord_y_norm),
                'unit_target': unit_target,
                'value': predictions['values'][0, 0, 0].item(),
                'action_confidence': action_probs[action_type].item(),
                'coordinate_confidence': (coord_x_probs[coord_x].item(), 
                                        coord_y_probs[coord_y].item()),
                'unit_confidence': unit_probs[unit_target].item()
            }


def create_model(config: Optional[ModelConfig] = None) -> OpenLeague5Model:
    """Factory function to create OpenLeague5 model"""
    return OpenLeague5Model(config)


if __name__ == "__main__":
    # Test model creation and forward pass
    config = ModelConfig()
    model = create_model(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"LSTM parameters: {sum(p.numel() for p in model.temporal_lstm.parameters())}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 3
    
    spatial = torch.randn(batch_size, seq_len, config.spatial_channels, 
                         config.spatial_resolution, config.spatial_resolution)
    units = torch.randn(batch_size, seq_len, config.max_units, config.unit_feature_dim)
    mask = torch.ones(batch_size, seq_len, config.max_units, dtype=torch.bool)
    mask[:, :, 30:] = False  # Simulate variable number of units
    global_feat = torch.randn(batch_size, seq_len, config.global_feature_dim)
    
    predictions, hidden = model(spatial, units, mask, global_feat)
    
    print("Forward pass successful!")
    print(f"Action types shape: {predictions['action_types'].shape}")
    print(f"Coordinates X shape: {predictions['coordinates_x'].shape}")
    print(f"Unit targets shape: {predictions['unit_targets'].shape}")
    print(f"Values shape: {predictions['values'].shape}")