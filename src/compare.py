"""Step 6: Compare baseline vs centralized vs federated, and draw the charts.

Reads results/experiment_log.csv and results/fl_rounds.csv and creates:
  results/comparison_by_junction.png  - bar chart of test MAE per junction
  results/fl_convergence.png          - federated MAE per round (convergence)
  results/fl_communication.png        - federated MAE vs megabytes sent
  results/comparison_table.md         - a table you can paste into the README
"""

import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENTS = {
    "baseline_last_hour": "Baseline (last hour)",
    "centralized_gru": "Centralized GRU",
    "federated_gru_fedavg": "Federated GRU (FedAvg)",
}

# 1. Load the experiment log and keep the MOST RECENT result for each experiment + junction
log = pd.read_csv("results/experiment_log.csv")
latest = log.groupby(["experiment", "junction"]).tail(1)
table = latest.pivot(index="junction", columns="experiment", values="mae")
table = table[[e for e in EXPERIMENTS if e in table.columns]]
table.loc["Average"] = table.mean()

print("Test MAE (vehicles per hour, lower is better):")
print(table.round(2).rename(columns=EXPERIMENTS).to_string())

# Save as a Markdown table for the README
with open("results/comparison_table.md", "w") as f:
    headers = ["Junction"] + [EXPERIMENTS[c] for c in table.columns]
    f.write("| " + " | ".join(headers) + " |\n")
    f.write("|" + "---|" * len(headers) + "\n")
    for junction, row in table.iterrows():
        f.write(f"| {junction} | " + " | ".join(f"{v:.2f}" for v in row) + " |\n")
print("\nSaved results/comparison_table.md")

# 2. Bar chart: each junction, one bar per experiment
ax = table.rename(columns=EXPERIMENTS).plot(kind="bar", figsize=(10, 5), rot=0)
ax.set_title("Test error by junction (lower is better)")
ax.set_xlabel("Junction")
ax.set_ylabel("MAE (vehicles per hour)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/comparison_by_junction.png", dpi=150)
print("Saved results/comparison_by_junction.png")

# 3. Convergence: federated average MAE after each round, vs the centralized result
rounds = pd.read_csv("results/fl_rounds.csv")
central_avg = table.loc["Average", "centralized_gru"] if "centralized_gru" in table else None

plt.figure(figsize=(9, 5))
plt.plot(rounds["round"], rounds["average_mae"], marker="o", label="Federated GRU (FedAvg)")
if central_avg is not None:
    plt.axhline(central_avg, color="gray", linestyle="--", label=f"Centralized GRU ({central_avg:.2f})")
plt.title("Federated learning convergence")
plt.xlabel("Round")
plt.ylabel("Average test MAE (vehicles per hour)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fl_convergence.png", dpi=150)
print("Saved results/fl_convergence.png")

# 4. Communication trade-off: how much accuracy we get per megabyte sent
plt.figure(figsize=(9, 5))
plt.plot(rounds["mb_sent_total"], rounds["average_mae"], marker="o")
plt.title("Accuracy vs communication cost")
plt.xlabel("Total data sent between server and clients (MB)")
plt.ylabel("Average test MAE (vehicles per hour)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/fl_communication.png", dpi=150)
print("Saved results/fl_communication.png")

# 5. Print a few key numbers for the README
best = rounds.loc[rounds["average_mae"].idxmin()]
print(f"\nBest federated round: {int(best['round'])} "
      f"(average MAE {best['average_mae']:.2f}, {best['mb_sent_total']:.2f} MB sent)")