"""Step 4: The GRU model plus helper functions to train and evaluate it.

Both the centralized script and the Flower (federated) scripts use this file,
so the model is exactly the same in every experiment - a fair comparison.
"""

import csv
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class TrafficGRU(nn.Module):
    """Reads the last 24 hours of traffic and predicts the next hour.

    GRU layer:    reads the 24 hours one by one and keeps a "memory" (hidden state)
    Linear layer: turns the final memory into one number - the prediction
    """

    def __init__(self, hidden_size=32):
        super().__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, 24 hours, 1 feature)
        all_steps, _ = self.gru(x)          # memory after every hour
        last_step = all_steps[:, -1, :]     # memory after the final hour
        return self.output_layer(last_step) # shape: (batch, 1)


def set_seed(seed=42):
    """Fix all random number generators so results are reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model, X, y, epochs=5, batch_size=64, learning_rate=0.001):
    """Train the model on (X, y) and return the average loss of the last epoch."""
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    # Shuffling the WINDOWS is fine: every window is already inside the training period.
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_function = nn.MSELoss()  # Mean Squared Error: punishes big mistakes
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()                       # 1. clear old gradients
            predictions = model(X_batch)                # 2. make predictions
            loss = loss_function(predictions, y_batch)  # 3. measure the error
            loss.backward()                             # 4. work out how to adjust each weight
            optimizer.step()                            # 5. adjust the weights
            total_loss += loss.item() * len(X_batch)
        average_loss = total_loss / len(dataset)
    return average_loss


def predict(model, X):
    """Return the model's predictions (still in scaled units)."""
    model.eval()
    with torch.no_grad():  # no learning here, so skip gradient bookkeeping
        return model(torch.from_numpy(X)).numpy()


def evaluate(model, X, y, scaler):
    """Return MAE and RMSE in REAL vehicle counts (not scaled units)."""
    predictions = scaler.inverse_transform(predict(model, X))
    actual = scaler.inverse_transform(y)
    return error_metrics(predictions, actual)


def baseline_last_hour(X, y, scaler):
    """Simple baseline: predict that next hour = this hour."""
    predictions = scaler.inverse_transform(X[:, -1, :])  # the last hour of each window
    actual = scaler.inverse_transform(y)
    return error_metrics(predictions, actual)


def error_metrics(predictions, actual):
    errors = predictions - actual
    mae = float(np.mean(np.abs(errors)))        # average size of the mistake
    rmse = float(np.sqrt(np.mean(errors ** 2))) # like MAE, but big mistakes count more
    return mae, rmse


def log_experiment(experiment, junction, mae, rmse, settings, path="results/experiment_log.csv"):
    """Append one result row to the experiment log (a CSV file)."""
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["timestamp", "experiment", "junction", "mae", "rmse", "settings"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            experiment, junction, round(mae, 3), round(rmse, 3), settings,
        ])