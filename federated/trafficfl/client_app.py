"""trafficfl: Flower ClientApp - one client = one road junction."""

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from trafficfl.task import junction_for_client, load_junction, model_tools

app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Receive the shared model, train it on THIS junction's data only, send back weights."""
    project_dir = context.run_config["project-dir"]
    tools = model_tools(project_dir)
    junction = junction_for_client(context)
    data = load_junction(project_dir, junction)

    # 1. Build the GRU and load the shared weights sent by the server
    model = tools.TrafficGRU(hidden_size=context.run_config["hidden-size"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    # 2. Train locally - the raw data never leaves this client
    tools.set_seed(context.run_config["seed"] + junction)
    train_loss = tools.train(
        model,
        data["X_train"],
        data["y_train"],
        epochs=context.run_config["local-epochs"],
        batch_size=context.run_config["batch-size"],
        learning_rate=msg.content["config"]["lr"],
    )

    # 3. Reply with ONLY the updated weights and a few numbers
    metrics = MetricRecord({
        "train_loss": float(train_loss),
        "num-examples": len(data["X_train"]),  # FedAvg uses this to weight the average
    })
    content = RecordDict({"arrays": ArrayRecord(model.state_dict()), "metrics": metrics})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Test the shared model on THIS junction's validation data."""
    project_dir = context.run_config["project-dir"]
    tools = model_tools(project_dir)
    junction = junction_for_client(context)
    data = load_junction(project_dir, junction)

    model = tools.TrafficGRU(hidden_size=context.run_config["hidden-size"])
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())

    val_mae, val_rmse = tools.evaluate(model, data["X_val"], data["y_val"], data["scaler"])

    metrics = MetricRecord({
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "num-examples": len(data["X_val"]),
    })
    return Message(content=RecordDict({"metrics": metrics}), reply_to=msg)