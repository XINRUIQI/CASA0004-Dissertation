"""Flat-feature-fusion baseline backtest kernel (M0..M4).

A single fair rolling-origin engine shared by every modality config; M0 is an
internal benchmark, M1-M4 differ only in which columns of the merged matrix are
selected. Used by scripts/run_baseline.py and scripts/sweep_baseline.py.
"""
