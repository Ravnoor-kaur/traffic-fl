"""trafficfl: Flower ServerApp - coordinates the rounds and averages the weights (FedAvg)."""

import csv
from pathlib import Path

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from trafficfl.task import JUNCTIONS, load_junction, model_tools

app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    project_dir = context.run_config["project-dir"]
    num_rounds = context.run_config["num-server-rounds"]
    hidden_size = context.run_config["hidden-size"]
    tools = model_tools(project_dir)
    results_dir = Path(project_dir) / "results"

    # 1. Create the starting model (same seed as the centralized experiment)
    tools.set_seed(context.run_config["seed"])
    global_model = tools.TrafficGRU(hidden_size=hidden_size)

    # Communication cost: each round, every client downloads AND uploads the whole model
    model_mb = sum(p.numel() * 4 for p in global_model.parameters()) / 1_000_000  # 4 bytes per float
    mb_per_round = model_mb * 2 * len(JUNCTIONS)
    print(f"Model size: {model_mb:.4f} MB -> {mb_per_round:.4f} MB sent per round")

    rounds_file = results_dir / "fl_rounds.csv"

    def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        """After each round, test the shared model on every junction's TEST set.

        This is for measuring the experiment only (to draw convergence charts);
        in a real deployment the server would not have anyone's test data.
        """
        model = tools.TrafficGRU(hidden_size=hidden_size)
        model.load_state_dict(arrays.to_torch_state_dict())

        maes, rmses = {}, {}
        for j in JUNCTIONS:
            d = load_junction(project_dir, j)
            maes[j], rmses[j] = tools.evaluate(model, d["X_test"], d["y_test"], d["scaler"])
        average_mae = sum(maes.values()) / len(maes)

        # Save one row per round -> used for the convergence / communication charts
        with open(rounds_file, "w" if server_round == 0 else "a", newline="") as f:
            writer = csv.writer(f)
            if server_round == 0:
                writer.writerow(["round", "mb_sent_total", "average_mae"]
                                + [f"mae_junction_{j}" for j in JUNCTIONS])
            writer.writerow([server_round, round(server_round * mb_per_round, 4),
                             round(average_mae, 3)] + [round(maes[j], 3) for j in JUNCTIONS])

        # After the final round, add the results to the shared experiment log
        if server_round == num_rounds:
            settings = (f"seed={context.run_config['seed']} rounds={num_rounds} "
                        f"local_epochs={context.run_config['local-epochs']} "
                        f"batch={context.run_config['batch-size']} "
                        f"lr={context.run_config['learning-rate']} hidden={hidden_size}")
            for j in JUNCTIONS:
                tools.log_experiment("federated_gru_fedavg", j, maes[j], rmses[j], settings,
                                     path=str(results_dir / "experiment_log.csv"))

        metrics = {"average_mae": average_mae}
        metrics.update({f"mae_junction_{j}": maes[j] for j in JUNCTIONS})
        return MetricRecord(metrics)

    # 2. FedAvg: every round, all 4 clients train and evaluate
    strategy = FedAvg(fraction_train=1.0, fraction_evaluate=1.0)

    # 3. Run the federated rounds
    result = strategy.start(
        grid=grid,
        initial_arrays=ArrayRecord(global_model.state_dict()),
        train_config=ConfigRecord({"lr": context.run_config["learning-rate"]}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # 4. Save the final shared model
    torch.save(result.arrays.to_torch_state_dict(), str(results_dir / "federated_gru.pt"))
    print(f"\nSaved results/federated_gru.pt and per-round results to {rounds_file}")