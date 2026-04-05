"""Tests for VMO:VL ratio analysis."""

from __future__ import annotations

import numpy as np

from vmo_sim.emg.vmo_vl_ratio import VMOVLRatioAnalyzer


def test_equal_ratio() -> None:
    analyzer = VMOVLRatioAnalyzer()
    signal = np.concatenate([np.zeros(20), np.ones(80) * 0.8])
    ratio = analyzer.compute_ratio(signal, signal)
    assert np.isclose(ratio, 1.0, atol=1e-3)


def test_weak_vmo() -> None:
    analyzer = VMOVLRatioAnalyzer()
    vmo = np.concatenate([np.zeros(20), np.ones(80) * 0.4])
    vl = np.concatenate([np.zeros(20), np.ones(80) * 0.8])
    ratio = analyzer.compute_ratio(vmo, vl)
    assert np.isclose(ratio, 0.5, atol=1e-3)


def test_onset_simultaneous() -> None:
    analyzer = VMOVLRatioAnalyzer()
    signal = np.concatenate([np.zeros(30), np.ones(70) * 0.8])
    timing_ms = analyzer.onset_timing(signal, signal, fs=1000.0, threshold=0.05)
    assert np.isclose(timing_ms, 0.0, atol=1e-6)
