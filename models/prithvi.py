"""
Loading prithvi and wrapping for finetuning
for different embeddings
"""

import timm
import torch
from torch import nn
from utils import *


class PrithviRegressor(nn.Module):
    def __init__(self, backbone: nn.Module, img_channels: int, num_outputs: int):
        """
        Prithvi classifier wrapper

        Args:
            backbone: prtihvi backbone loaded from terratorch
            img_channels: Number of input channels in your data
            num_outputs: Number of continous values to predict (1 val)
        """
        super().__init__()
        self.backbone = backbone
        self.num_outputs = num_outputs
        self.img_channels = img_channels

        # Channel adapter if not taking 6-channel input that prithvi expects
        self.channel_adapter = channel_adapter(6, self.img_channels)

        self.embed_dim = 192        # hardcoded for prtihvi tiny

        # Regression head
        self.regressor = nn.Linear(self.embed_dim, num_outputs)

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor [B, C, H, W]

        Returns:
            Class logits [B, num_classes]
        """
        # Adapt channels
        x = self.channel_adapter(x)

        # Get features from backbone
        features = self.backbone(x)

        # Get last layers features
        features = features[-1]

        # Extract CLS token (first token)
        cls_token = features[:, 0]

        # Classify
        logits = self.classifier(cls_token)  # [B, num_classes]

        return logits