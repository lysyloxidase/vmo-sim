"""Recursive EMG-to-activation dynamics."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class EMGToActivation:
    """Recursive filter converting processed EMG to muscle activation."""

    def __init__(
        self,
        electromechanical_delay: float = 0.040,
        shape_factor: float = -3.0,
        fs: float = 1000.0,
    ) -> None:
        self.electromechanical_delay = electromechanical_delay
        self.shape_factor = shape_factor
        self.fs = fs
        self.delay_samples = max(int(round(electromechanical_delay * fs)), 0)
        damping = 0.8
        self.beta1 = -2.0 * damping
        self.beta2 = damping**2
        self.alpha = 1.0 + self.beta1 + self.beta2

    def _shape_function(self, excitation: FloatArray) -> FloatArray:
        clipped = np.clip(excitation, 0.0, 1.0)
        if np.isclose(self.shape_factor, 0.0):
            return clipped
        numerator = np.exp(self.shape_factor * clipped) - 1.0
        denominator = np.exp(self.shape_factor) - 1.0
        return np.asarray(numerator / denominator, dtype=np.float64)

    def process(self, emg_envelope: np.ndarray) -> FloatArray:
        """Convert an EMG envelope into a smooth activation estimate."""

        envelope = np.asarray(emg_envelope, dtype=np.float64)
        delayed = np.zeros_like(envelope)
        if self.delay_samples == 0:
            delayed = envelope.copy()
        elif self.delay_samples < envelope.shape[-1]:
            delayed[..., self.delay_samples :] = envelope[..., : -self.delay_samples]

        activation_state = np.zeros_like(delayed)
        for sample_index in range(delayed.shape[-1]):
            prev1 = (
                activation_state[..., sample_index - 1] if sample_index >= 1 else 0.0
            )
            prev2 = (
                activation_state[..., sample_index - 2] if sample_index >= 2 else 0.0
            )
            activation_state[..., sample_index] = (
                self.alpha * delayed[..., sample_index]
                - self.beta1 * prev1
                - self.beta2 * prev2
            )

        return self._shape_function(np.clip(activation_state, 0.0, 1.0))


class EMGActivationDynamics(EMGToActivation):
    """Backward-compatible alias for EMG activation dynamics."""

    def __init__(self, alpha: float = 0.04, beta: float = 0.8) -> None:
        del alpha, beta
        super().__init__()
