"""Unit tests for the data preparation pipeline. Run with: pytest"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Let the tests import files from the src/ folder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_prep import Scaler, make_windows, split_by_time


def test_make_windows_creates_correct_examples():
    series = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    X, y = make_windows(series, window_size=3)

    assert X.shape == (2, 3, 1)  # 2 examples, 3 time steps, 1 feature
    assert y.shape == (2, 1)
    assert X[0].flatten().tolist() == [1, 2, 3]
    assert y[0, 0] == 4
    assert X[1].flatten().tolist() == [2, 3, 4]
    assert y[1, 0] == 5


def test_make_windows_rejects_series_that_is_too_short():
    with pytest.raises(ValueError):
        make_windows(np.array([1, 2, 3], dtype=np.float32), window_size=3)


def test_split_by_time_keeps_order_and_loses_nothing():
    series = np.arange(100)
    train, val, test = split_by_time(series, train_fraction=0.7, val_fraction=0.1)

    assert len(train) == 70 and len(val) == 10 and len(test) == 20
    # No shuffling: every training value comes before every test value
    assert train.max() < val.min() and val.max() < test.min()
    # Nothing lost or duplicated
    assert np.array_equal(np.concatenate([train, val, test]), series)


def test_scaler_round_trip_returns_original_values():
    values = np.array([10, 20, 30, 40], dtype=np.float32)
    scaler = Scaler().fit(values)
    scaled = scaler.transform(values)

    assert abs(scaled.mean()) < 1e-6          # scaled mean is 0
    assert abs(scaled.std() - 1) < 1e-6       # scaled std is 1
    assert np.allclose(scaler.inverse_transform(scaled), values)


def test_scaler_handles_constant_values():
    scaler = Scaler().fit(np.array([5, 5, 5], dtype=np.float32))
    assert scaler.std == 1.0  # no divide-by-zero