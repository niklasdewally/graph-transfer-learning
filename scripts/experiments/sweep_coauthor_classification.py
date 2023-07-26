import datetime
import wandb
import sys

N = 75

sweep_config = {
    "metric": {"goal": "maximize", "name": "small-to-large-accuracy"},
    "method": "random",
    "parameters": {
        "n_runs": {"value": 5},
        "batch_size": {"values": [16, 32, 64, 128]},
        "hidden_layers": {"min": 16, "max": 256},
        "lr": {"min": 0.0001, "max": 0.1},
        "k": {"min": 2, "max": 5},
        "patience": {"value": 25},
        "min_delta": {"value": 0.01},
        "epochs": {"value": 100},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3},
    "run_cap": N,
}

current_date_time: str = datetime.datetime.now().strftime("%Y-%m-%d (T%H%M)")


def main():
    from coauthor_classification import do_run

    if len(sys.argv) > 1 and sys.argv[1] is not None:
        sweep_id = sys.argv[1]
        wandb.agent(sweep_id=sweep_id, function=lambda: train(do_run))
        return

    for model in ["graphsage", "egi", "triangle"]:
        sweep_config.update({"name": f"{model} ({current_date_time})"}),
        sweep_config["parameters"].update({"model": {"value": model}})

        sweep_id = wandb.sweep(
            sweep=sweep_config, project="Coauthor Classification Sweeps"
        )

        wandb.agent(sweep_id=sweep_id, function=lambda: train(do_run))


def train(do_run) -> None:
    wandb.init()

    # to reduce variance, do many runs, and optimise on the average
    for i in range(wandb.config["n_runs"]):
        wandb.define_metric("small-to-large-accuracy", summary="mean")
        do_run(False)

    wandb.finish()


if __name__ == "__main__":
    main()
