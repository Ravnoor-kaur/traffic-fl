"""Step 3: Data preparation.

Turns the raw traffic.csv into training examples for each junction:
  1. Load and sort the data
  2. Check for gaps in the hourly timeline
  3. Split each junction's data by time (train / validation / test)
  4. Scale the numbers using ONLY the training data
  5. Cut the timeline into sliding windows (last 24 hours -> next hour)
"""

import numpy as np
import pandas as pd

WINDOW_SIZE = 24      # use the last 24 hours as input
TRAIN_FRACTION = 0.7  # first 70% of time for training
VAL_FRACTION = 0.1    # next 10% for validation; the last 20% is the test set


def load_data(path="data/traffic.csv"):
    """Read the CSV, convert dates, and sort by junction and time."""
    df = pd.read_csv(path)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.sort_values(["Junction", "DateTime"]).reset_index(drop=True)
    return df


def get_junction_series(df, junction):
    """Return one junction's hourly vehicle counts as a numpy array (oldest first)."""
    junction_rows = df[df["Junction"] == junction]
    return junction_rows["Vehicles"].to_numpy(dtype=np.float32)


def count_missing_hours(df, junction):
    """Count hours missing from a junction's timeline (should be 0)."""
    times = df[df["Junction"] == junction]["DateTime"]
    expected_hours = pd.date_range(times.min(), times.max(), freq="h")
    return len(expected_hours) - len(times)


def split_by_time(series, train_fraction=TRAIN_FRACTION, val_fraction=VAL_FRACTION):
    """Split a timeline into train / val / test WITHOUT shuffling.

    The model must never see the future during training (no data leakage),
    so the oldest data is for training and the newest is for testing.
    """
    n = len(series)
    train_end = int(n * train_fraction)
    val_end = int(n * (train_fraction + val_fraction))
    return series[:train_end], series[train_end:val_end], series[val_end:]


class Scaler:
    """Standard scaling: (value - mean) / std.

    fit() must only be called on TRAINING data. Using the test data's
    mean/std would leak information about the future into training.
    """

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, values):
        self.mean = float(np.mean(values))
        self.std = float(np.std(values))
        if self.std == 0:
            self.std = 1.0  # avoid dividing by zero if all values are equal
        return self

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        """Turn scaled numbers back into real vehicle counts."""
        return values * self.std + self.mean


def make_windows(series, window_size=WINDOW_SIZE):
    """Cut a timeline into (input, target) training examples.

    Example with window_size=3 and series [1, 2, 3, 4, 5]:
        X = [[1, 2, 3],      y = [4,
             [2, 3, 4]]           5]
    X gets shape (examples, window_size, 1) because the GRU expects
    one feature (vehicle count) per time step.
    """
    if len(series) <= window_size:
        raise ValueError("Series is too short for the chosen window size.")
    X, y = [], []
    for start in range(len(series) - window_size):
        X.append(series[start:start + window_size])
        y.append(series[start + window_size])
    X = np.array(X, dtype=np.float32)[:, :, np.newaxis]
    y = np.array(y, dtype=np.float32)[:, np.newaxis]
    return X, y


def prepare_junction(df, junction, window_size=WINDOW_SIZE):
    """Run the full pipeline for one junction (= one federated client)."""
    series = get_junction_series(df, junction)
    train, val, test = split_by_time(series)

    scaler = Scaler().fit(train)  # learn mean/std from training data only

    data = {"scaler": scaler}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        X, y = make_windows(scaler.transform(part), window_size)
        data[f"X_{name}"] = X
        data[f"y_{name}"] = y
    return data


if __name__ == "__main__":
    # Running this file directly prints a summary of the prepared data
    df = load_data()
    for junction in sorted(df["Junction"].unique()):
        data = prepare_junction(df, junction)
        print(
            f"Junction {junction}: missing hours = {count_missing_hours(df, junction)}, "
            f"train = {len(data['X_train'])}, val = {len(data['X_val'])}, "
            f"test = {len(data['X_test'])} windows, "
            f"mean = {data['scaler'].mean:.1f}, std = {data['scaler'].std:.1f}"
        )