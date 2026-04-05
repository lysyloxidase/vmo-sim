"""Clinical analysis of the VMO:VL activation ratio."""

from __future__ import annotations

import numpy as np


class VMOVLRatioAnalyzer:
    """Compute and analyze the clinically relevant VMO:VL activation ratio."""

    @staticmethod
    def _contraction_mask(vmo_emg: np.ndarray, vl_emg: np.ndarray) -> np.ndarray:
        activity = 0.5 * (np.abs(vmo_emg) + np.abs(vl_emg))
        threshold = max(0.05 * float(np.max(activity)), 1e-8)
        mask = activity >= threshold
        if np.any(mask):
            return np.asarray(mask, dtype=bool)
        return np.asarray(np.ones_like(activity, dtype=bool))

    def compute_ratio(self, vmo_emg: np.ndarray, vl_emg: np.ndarray) -> float:
        """Compute the mean VMO:VL amplitude ratio over the active period."""

        vmo = np.asarray(vmo_emg, dtype=np.float64)
        vl = np.asarray(vl_emg, dtype=np.float64)
        if vmo.shape != vl.shape:
            raise ValueError("vmo_emg and vl_emg must have the same shape.")
        mask = self._contraction_mask(vmo, vl)
        vmo_mean = float(np.mean(np.abs(vmo[mask])))
        vl_mean = float(np.mean(np.abs(vl[mask])))
        return vmo_mean / max(vl_mean, 1e-8)

    def onset_timing(
        self,
        vmo_emg: np.ndarray,
        vl_emg: np.ndarray,
        fs: float = 1000.0,
        threshold: float = 0.05,
    ) -> float:
        """Return VMO onset time minus VL onset time in milliseconds."""

        vmo = np.asarray(vmo_emg, dtype=np.float64)
        vl = np.asarray(vl_emg, dtype=np.float64)
        if vmo.shape != vl.shape:
            raise ValueError("vmo_emg and vl_emg must have the same shape.")

        def onset_index(signal: np.ndarray) -> int:
            above = np.flatnonzero(signal >= threshold)
            return int(above[0]) if above.size > 0 else signal.shape[-1] - 1

        vmo_onset = onset_index(vmo)
        vl_onset = onset_index(vl)
        return 1000.0 * (vmo_onset - vl_onset) / fs

    def classify(self, ratio: float, timing_ms: float) -> str:
        """Classify VMO:VL function from amplitude and onset timing."""

        if ratio >= 0.9 and timing_ms <= 10.0:
            return "normal"
        if ratio >= 0.8 and timing_ms <= 20.0:
            return "mild_dysfunction"
        if ratio >= 0.6 and timing_ms <= 40.0:
            return "moderate_dysfunction"
        return "severe"


def compute_vmo_vl_ratio(vmo_signal: np.ndarray, vl_signal: np.ndarray) -> float:
    """Backward-compatible wrapper for VMO:VL ratio computation."""

    return VMOVLRatioAnalyzer().compute_ratio(vmo_signal, vl_signal)
