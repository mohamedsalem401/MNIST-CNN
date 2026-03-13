from __future__ import annotations

from mnist_hybrid.config import ModelConfig
from mnist_hybrid.models.base import IntervenableModel
from mnist_hybrid.models.cnn import SmallCNNClassifier
from mnist_hybrid.models.mlp import MLPClassifier


def build_model(config: ModelConfig) -> IntervenableModel:
    if config.architecture == "mlp":
        return MLPClassifier(
            hidden_sizes=list(config.hidden_sizes),
            num_classes=config.num_classes,
            dropout=config.dropout,
            extra_parametric_layer=config.extra_parametric_layer,
            growth_enabled=config.growth_enabled,
        )

    if config.architecture == "cnn":
        return SmallCNNClassifier(
            channels=list(config.cnn_channels),
            latent_dim=config.latent_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
            extra_parametric_layer=config.extra_parametric_layer,
        )

    raise ValueError(f"Unknown architecture: {config.architecture}")
