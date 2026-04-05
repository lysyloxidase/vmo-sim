"""Standard EMG signal preprocessing pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal as scipy_signal  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]


class EMGPreprocessor:
    """Standard EMG signal preprocessing pipeline."""

    def __init__(self, line_frequency: float = 50.0) -> None:
        self.line_frequency = line_frequency

    @staticmethod
    def _as_float_array(signal: np.ndarray) -> FloatArray:
        return np.asarray(signal, dtype=np.float64)

    def bandpass_filter(
        self,
        signal: np.ndarray,
        fs: float = 1000.0,
        low: float = 20.0,
        high: float = 500.0,
        order: int = 4,
    ) -> FloatArray:
        """Apply a Butterworth bandpass filter to raw EMG."""

        safe_high = min(high, 0.99 * fs / 2.0)
        if not 0.0 < low < safe_high < fs / 2.0:
            raise ValueError("Bandpass frequencies must satisfy 0 < low < high < fs/2.")
        emg = self._as_float_array(signal)
        sos = scipy_signal.butter(
            order,
            [low, safe_high],
            btype="bandpass",
            fs=fs,
            output="sos",
        )
        return np.asarray(scipy_signal.sosfiltfilt(sos, emg, axis=-1), dtype=np.float64)

    def notch_filter(
        self,
        signal: np.ndarray,
        fs: float = 1000.0,
        freq: float = 50.0,
        q: float = 30.0,
    ) -> FloatArray:
        """Apply a notch filter for line-noise removal."""

        if not 0.0 < freq < fs / 2.0:
            raise ValueError("Notch frequency must satisfy 0 < freq < fs/2.")
        emg = self._as_float_array(signal)
        b, a = scipy_signal.iirnotch(freq, q, fs)
        return np.asarray(scipy_signal.filtfilt(b, a, emg, axis=-1), dtype=np.float64)

    def rectify(self, signal: np.ndarray) -> FloatArray:
        """Apply full-wave rectification."""

        emg = self._as_float_array(signal)
        return np.abs(emg)

    def envelope(
        self,
        signal: np.ndarray,
        fs: float = 1000.0,
        cutoff: float = 6.0,
    ) -> FloatArray:
        """Extract a smooth linear envelope with low-pass filtering."""

        if not 0.0 < cutoff < fs / 2.0:
            raise ValueError("Envelope cutoff must satisfy 0 < cutoff < fs/2.")
        emg = self._as_float_array(signal)
        sos = scipy_signal.butter(4, cutoff, btype="lowpass", fs=fs, output="sos")
        return np.asarray(scipy_signal.sosfiltfilt(sos, emg, axis=-1), dtype=np.float64)

    def normalize_mvc(self, signal: np.ndarray, mvc: float) -> FloatArray:
        """Normalize processed EMG by a maximum voluntary contraction value."""

        if mvc <= 0.0:
            raise ValueError("mvc must be positive.")
        emg = self._as_float_array(signal)
        return emg / mvc

    def full_pipeline(
        self,
        raw_emg: np.ndarray,
        fs: float = 1000.0,
        mvc: float | None = None,
    ) -> FloatArray:
        """Apply the full preprocessing pipeline to raw EMG."""

        filtered = self.bandpass_filter(raw_emg, fs=fs)
        denoised = self.notch_filter(filtered, fs=fs, freq=self.line_frequency)
        rectified = self.rectify(denoised)
        emg_envelope = self.envelope(rectified, fs=fs)
        if mvc is not None:
            return self.normalize_mvc(emg_envelope, mvc)
        return emg_envelope


def bandpass_filter(
    signal: np.ndarray,
    low_hz: float,
    high_hz: float,
    sample_rate_hz: float,
    order: int = 4,
) -> FloatArray:
    """Functional wrapper for bandpass EMG filtering."""

    return EMGPreprocessor().bandpass_filter(
        signal,
        fs=sample_rate_hz,
        low=low_hz,
        high=high_hz,
        order=order,
    )


def rectify(signal: np.ndarray) -> FloatArray:
    """Functional wrapper for full-wave rectification."""

    return EMGPreprocessor().rectify(signal)


def linear_envelope(
    signal: np.ndarray,
    cutoff_hz: float,
    sample_rate_hz: float,
) -> FloatArray:
    """Functional wrapper for linear envelope extraction."""

    return EMGPreprocessor().envelope(signal, fs=sample_rate_hz, cutoff=cutoff_hz)


def normalize_emg(signal: np.ndarray, reference_value: float) -> FloatArray:
    """Functional wrapper for MVC normalization."""

    return EMGPreprocessor().normalize_mvc(signal, mvc=reference_value)


def preprocess_pipeline(
    signal: np.ndarray,
    sample_rate_hz: float,
    low_hz: float = 20.0,
    high_hz: float = 450.0,
    envelope_cutoff_hz: float = 6.0,
    mvc: float | None = None,
) -> FloatArray:
    """Functional wrapper for the full preprocessing chain."""

    processor = EMGPreprocessor()
    filtered = processor.bandpass_filter(
        signal, fs=sample_rate_hz, low=low_hz, high=high_hz
    )
    denoised = processor.notch_filter(
        filtered, fs=sample_rate_hz, freq=processor.line_frequency
    )
    rectified_signal = processor.rectify(denoised)
    emg_envelope = processor.envelope(
        rectified_signal, fs=sample_rate_hz, cutoff=envelope_cutoff_hz
    )
    if mvc is not None:
        return processor.normalize_mvc(emg_envelope, mvc=mvc)
    return emg_envelope
