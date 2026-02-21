"""
CNN-LSTM Model for Drowsy Driver Detection

This model uses:
- MobileNetV2 as CNN backbone for spatial feature extraction
- LSTM for temporal modeling across image sequences
- Binary classification for drowsy vs not-drowsy detection
"""

import torch
import torch.nn as nn
from torchvision import models


class CNN_LSTM(nn.Module):
    """
    Combined CNN-LSTM architecture for sequence-based drowsiness detection.
    
    Args:
        num_classes (int): Number of output classes (default: 2 for drowsy/not-drowsy)
        hidden_size (int): LSTM hidden state size (default: 128)
        num_layers (int): Number of LSTM layers (default: 1)
        dropout (float): Dropout rate for regularization (default: 0.3)
        freeze_cnn (bool): Whether to freeze CNN backbone weights (default: False)
    """
    
    def __init__(
        self, 
        num_classes=2, 
        hidden_size=128, 
        num_layers=1,
        dropout=0.3,
        freeze_cnn=False
    ):
        super(CNN_LSTM, self).__init__()
        
        # CNN backbone - MobileNetV2 (lightweight and efficient)
        weights = models.MobileNet_V2_Weights.DEFAULT
        mobilenet = models.mobilenet_v2(weights=weights)
        self.cnn = mobilenet.features
        
        # Optionally freeze CNN weights
        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
        
        # Adaptive pooling to get fixed size output
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # MobileNetV2 output channels
        self.cnn_output_size = 1280
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Classifier head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
        # Store hyperparameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, C, H, W)
                where:
                - batch: batch size
                - seq_len: number of frames in sequence
                - C: number of channels (3 for RGB)
                - H, W: image height and width
        
        Returns:
            torch.Tensor: Output logits of shape (batch, num_classes)
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # Reshape to process all frames through CNN
        # (batch * seq_len, C, H, W)
        x = x.view(batch_size * seq_len, C, H, W)
        
        # Extract spatial features with CNN
        x = self.cnn(x)
        x = self.pool(x)
        
        # Reshape back to sequence format
        # (batch, seq_len, cnn_output_size)
        x = x.view(batch_size, seq_len, -1)
        
        # Process temporal sequence with LSTM
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last time step output for classification
        # (batch, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout and classify
        x = self.dropout(last_output)
        out = self.fc(x)
        
        return out
    
    def get_attention_weights(self, x):
        """
        Get attention weights for each frame in the sequence.
        Useful for interpretability.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, C, H, W)
        
        Returns:
            torch.Tensor: Attention weights of shape (batch, seq_len)
        """
        batch_size, seq_len, C, H, W = x.size()
        
        # Extract features
        x = x.view(batch_size * seq_len, C, H, W)
        x = self.cnn(x)
        x = self.pool(x)
        x = x.view(batch_size, seq_len, -1)
        
        # Get LSTM outputs
        lstm_out, _ = self.lstm(x)
        
        # Simple attention mechanism
        # Calculate importance score for each time step
        attention_scores = torch.mean(lstm_out ** 2, dim=2)  # (batch, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=1)
        
        return attention_weights


def create_cnn_lstm(
    num_classes=2,
    hidden_size=128,
    num_layers=1,
    dropout=0.3,
    freeze_cnn=False,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Factory function to create CNN-LSTM model.
    
    Args:
        num_classes (int): Number of output classes
        hidden_size (int): LSTM hidden size
        num_layers (int): Number of LSTM layers
        dropout (float): Dropout rate
        freeze_cnn (bool): Whether to freeze CNN weights
        device (str): Device to place model on
    
    Returns:
        CNN_LSTM: Initialized model on specified device
    """
    model = CNN_LSTM(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        freeze_cnn=freeze_cnn
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel created on device: {device}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing CNN-LSTM model...")
    
    # Create model
    model = create_cnn_lstm(
        num_classes=2,
        hidden_size=128,
        num_layers=1,
        dropout=0.3
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 10
    channels = 3
    height = 224
    width = 224
    
    dummy_input = torch.randn(batch_size, seq_len, channels, height, width)
    
    print(f"\nInput shape: {dummy_input.shape}")
    
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test attention weights
    attention = model.get_attention_weights(dummy_input)
    print(f"Attention weights shape: {attention.shape}")
    print(f"Sample attention weights: {attention[0].detach().numpy()}")
    
    print("\n✓ Model test passed!")
