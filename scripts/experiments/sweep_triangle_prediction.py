import sys

import wandb
import datetime

MAX_RUNS_PER_MODEL = 50
MODELS = ["graphsage", "egi", "triangle"]

sweep_config = {
    "metric": {"goal": "maximize", "name": "acc"},
    "method": "random",
    "parameters": {
        "n_runs": {"value": 5},
        "batch_size": {"values": [16, 32, 64, 128, 256]},
        "hidden_layers": {"min": 16, "max": 512},
        "lr": {"min": 0.0001, "max": 0.1},
        "k": {"min": 2, "max": 5},
        "patience": {"value": 25},
        "min_delta": {"value": 0.01},
        "epochs": {"value": 100},
        "source_size": {"value": 10000},
        "target_size": {"value": 10000},
    },
    "early_terminate": {"type": "hyperband", "min_iter": 3},
    "run_cap": MAX_RUNS_PER_MODEL,
}


current_date_time: str = datetime.datetime.now().strftime("%Y-%m-%d (T%H%M)")


def main() -> int:
    from aug_triangle_prediction import do_run

    if len(sys.argv) > 1 and sys.argv[1] is not None:
        sweep_id = sys.argv[1]
        wandb.agent(sweep_id=sweep_id, function=lambda: train(do_run))
        return 0

    for model in ["graphsage", "egi", "triangle"]:
        sweep_config.update({"name": f"{model} ({current_date_time})"}),
        sweep_config["parameters"].update({"model": {"value": model}})

        sweep_id = wandb.sweep(
            sweep=sweep_config, project="Aug 2023 Triangle Prediction Sweeps"
        )

        wandb.agent(sweep_id=sweep_id, function=lambda: train(do_run))

    return 0


def train(do_run) -> None:
    wandb.init()

    # to reduce variance, do many runs, and optimise on the average
    wandb.define_metric("acc", summary="mean")
    for i in range(wandb.config["n_runs"]):
        do_run(eval_mode="validate")

    wandb.finish()


if __name__ == "__main__":
    sys.exit(main())
