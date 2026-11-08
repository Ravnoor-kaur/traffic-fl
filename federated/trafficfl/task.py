"""trafficfl: shared helpers for the Flower ClientApp and ServerApp.

Instead of copying our code, this file reuses src/data_prep.py and src/model.py,
so the federated experiment uses EXACTLY the same data pipeline and GRU model
as the centralized experiment.
"""

import sys
from functools import lru_cache
from pathlib import Path

JUNCTIONS = [1, 2, 3, 4]


def use_project_code(project_dir):
    """Let Python import files from the project's src/ folder."""
    src_folder = str(Path(project_dir) / "src")
    if src_folder not in sys.path:
        sys.path.insert(0, src_folder)


@lru_cache(maxsize=None)
def load_junction(project_dir, junction):
    """Prepare ONE junction's data - all that a single client may see.

    The result is cached, so later rounds reuse it instead of reloading the CSV.
    (In this simulation all junctions live in one CSV file; in a real deployment
    each junction would hold its own file on its own machine.)
    """
    use_project_code(project_dir)
    from data_prep import load_data, prepare_junction

    df = load_data(str(Path(project_dir) / "data" / "traffic.csv"))
    return prepare_junction(df, junction)


def model_tools(project_dir):
    """Return our src/model.py module (TrafficGRU, train, evaluate, ...)."""
    use_project_code(project_dir)
    import model

    return model


def junction_for_client(context):
    """Map Flower's client number (0-3) to a junction (1-4)."""
    num_partitions = context.node_config["num-partitions"]
    if num_partitions != len(JUNCTIONS):
        raise ValueError(
            f"Expected 4 simulated clients (one per junction) but Flower started "
            f"{num_partitions}. Run with: flwr run . --federation-config \"num-supernodes=4\""
        )
    return JUNCTIONS[context.node_config["partition-id"]]