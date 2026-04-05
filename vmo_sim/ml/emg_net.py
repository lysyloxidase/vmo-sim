"""Neural networks for EMG-to-force prediction."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn


class EMGForceNet(nn.Module):
    """1D CNN for predicting normalized muscle force from EMG."""

    def __init__(self, n_channels: int = 1, output_dim: int = 1) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.output_dim = output_dim
        self.input_layout = "channels_first"
        self.features = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, emg_signal: torch.Tensor) -> torch.Tensor:
        """Predict normalized force from channels-first EMG input."""

        features = self.features(emg_signal)
        return cast(torch.Tensor, self.regressor(features))


class EMGForceLSTM(nn.Module):
    """Bidirectional LSTM for temporal EMG-to-force mapping."""

    def __init__(self, n_channels: int = 1, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.input_layout = "sequence_first"
        self.lstm = nn.LSTM(
            input_size=n_channels,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(nn.Linear(2 * hidden_dim, 1), nn.Sigmoid())

    def forward(self, emg_signal: torch.Tensor) -> torch.Tensor:
        """Predict normalized force from sequence-first EMG input."""

        _, (hidden_state, _) = self.lstm(emg_signal)
        top_layer_forward = hidden_state[-2]
        top_layer_backward = hidden_state[-1]
        combined = torch.cat([top_layer_forward, top_layer_backward], dim=-1)
        return cast(torch.Tensor, self.head(combined))
