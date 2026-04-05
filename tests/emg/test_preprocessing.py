"""Tests for EMG preprocessing."""

from __future__ import annotations

import numpy as np

from vmo_sim.emg.preprocessing import EMGPreprocessor


def _fft_amplitude(signal: np.ndarray, fs: float, freq: float) -> float:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, d=1.0 / fs)
    index = int(np.argmin(np.abs(freqs - freq)))
    return float(np.abs(spectrum[index]))


def test_bandpass() -> None:
    fs = 2000.0
    time = np.arange(0.0, 1.0, 1.0 / fs)
    raw = (
        0.8 * np.sin(2.0 * np.pi * 10.0 * time)
        + 1.0 * np.sin(2.0 * np.pi * 80.0 * time)
        + 0.8 * np.sin(2.0 * np.pi * 600.0 * time)
    )
    filtered = EMGPreprocessor().bandpass_filter(raw, fs=fs, low=20.0, high=450.0)
    amp_80 = _fft_amplitude(filtered, fs, 80.0)
    amp_10 = _fft_amplitude(filtered, fs, 10.0)
    amp_600 = _fft_amplitude(filtered, fs, 600.0)
    assert amp_80 > 5.0 * amp_10
    assert amp_80 > 5.0 * amp_600


def test_envelope() -> None:
    fs = 1000.0
    time = np.arange(0.0, 1.0, 1.0 / fs)
    carrier = np.abs(np.sin(2.0 * np.pi * 80.0 * time))
    modulated = carrier * (0.3 + 0.7 * np.sin(2.0 * np.pi * 2.0 * time) ** 2)
    processor = EMGPreprocessor()
    envelope = processor.envelope(modulated, fs=fs, cutoff=6.0)
    assert envelope.shape == modulated.shape
    assert np.mean(np.abs(np.diff(envelope))) < np.mean(np.abs(np.diff(modulated)))


def test_pipeline_shape() -> None:
    fs = 1000.0
    time = np.arange(0.0, 1.0, 1.0 / fs)
    raw = np.sin(2.0 * np.pi * 80.0 * time)
    processed = EMGPreprocessor().full_pipeline(raw, fs=fs, mvc=1.0)
    assert processed.shape == raw.shape
