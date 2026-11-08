"""Step 4: Centralized training (the normal way).

All 4 junctions' training data is put in one place and one GRU is trained on it.
This is the "best case" we will compare federated learning against.
"""

import time

import numpy as np
import torch

from data_prep import load_data, prepare_junction
from model import TrafficGRU, baseline_last_hour, evaluate, log_experiment, set_seed, train

SEED = 42
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32


def main():
    set_seed(SEED)

    # 1. Prepare each junction separately (each scaled with its own training mean/std)
    df = load_data()
    junctions = sorted(df["Junction"].unique())
    data = {j: prepare_junction(df, j) for j in junctions}

    # 2. Centralized = combine everyone's training windows into one big pile
    X_train = np.concatenate([data[j]["X_train"] for j in junctions])
    y_train = np.concatenate([data[j]["y_train"] for j in junctions])
    print(f"Training on {len(X_train)} windows from {len(junctions)} junctions")

    # 3. Train, printing validation error after each epoch so we can watch it learn
    model = TrafficGRU(hidden_size=HIDDEN_SIZE)
    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        train_loss = train(model, X_train, y_train, epochs=1,
                           batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE)
        val_maes = [evaluate(model, data[j]["X_val"], data[j]["y_val"], data[j]["scaler"])[0]
                    for j in junctions]
        print(f"Epoch {epoch:2d} | train loss {train_loss:.4f} | "
              f"average validation MAE {np.mean(val_maes):.2f} vehicles")
    print(f"Training took {time.time() - start:.1f} seconds")

    # 4. Final exam: test set, per junction, compared with the simple baseline
    settings = f"seed={SEED} epochs={EPOCHS} batch={BATCH_SIZE} lr={LEARNING_RATE} hidden={HIDDEN_SIZE}"
    print("\nTest results (errors in vehicles per hour, lower is better):")
    print(f"{'Junction':<10}{'Baseline MAE':>14}{'GRU MAE':>10}{'GRU RMSE':>10}")
    for j in junctions:
        d = data[j]
        base_mae, base_rmse = baseline_last_hour(d["X_test"], d["y_test"], d["scaler"])
        mae, rmse = evaluate(model, d["X_test"], d["y_test"], d["scaler"])
        print(f"{j:<10}{base_mae:>14.2f}{mae:>10.2f}{rmse:>10.2f}")
        log_experiment("baseline_last_hour", j, base_mae, base_rmse, "none")
        log_experiment("centralized_gru", j, mae, rmse, settings)

    torch.save(model.state_dict(), "results/central_gru.pt")
    print("\nSaved model to results/central_gru.pt and results to results/experiment_log.csv")


if __name__ == "__main__":
    main()