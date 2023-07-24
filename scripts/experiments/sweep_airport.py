"""
A wandb sweep of a simple airport classification experiment, without transfer learning.
Used to tune hyper-parameters of each model.

For more information on sweeping hyperparameters with wandb, see:
https://docs.wandb.ai/guides/sweep/
"""

import argparse
import datetime
import pathlib
import tempfile

import torch

import wandb


# setup directorys to use for airport data
SCRIPT_DIR: pathlib.Path = pathlib.Path(__file__).parent.resolve()
PROJECT_DIR: pathlib.Path = SCRIPT_DIR.parent.resolve()
DATA_DIR: pathlib.Path = PROJECT_DIR / "data" / "airports"

# directory to store temporary model weights used while training
TMP_DIR: tempfile.TemporaryDirectory[str] = tempfile.TemporaryDirectory()

current_date_time: str = datetime.datetime.now().strftime("%Y-%m-%d (T%H%M)")
sweep_config = {
    "project": "Airport hyperparams sweep",
    "entity": "sta-graph-transfer-learning",
    "metric": {"goal": "maximize", "name": "target-classification-accuracy"},
    "method": "bayes",
    "parameters": {
        "n_runs": {"value": 5},
        "batch_size": {"values": [16, 32, 64]},
        "hidden_layers": {"min": 16, "max": 128},
        "lr": {"min": 0.0001, "max": 0.1},
        "k": {"min": 1, "max": 5},
        "patience": {"value": 20},
        "min_delta": {"value": 0.01},
        "epochs": {"value": 200},
    },
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    # read model type from arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["egi", "triangle", "graphsage"])
    parser.add_argument("--sweep-id", default=None, required=False)

    args = parser.parse_args()
    model = args.model
    sweep_id = args.sweep_id

    from airport_direct_transfer import do_run

    if sweep_id is None:
        # add model and name to sweep
        sweep_config.update({"name": f"{model} ({current_date_time})"}),
        sweep_config["parameters"].update({"model": {"value": model}})

        sweep_id = wandb.sweep(sweep=sweep_config)
    wandb.agent(sweep_id=sweep_id, function=lambda: train(do_run))


def train(do_run) -> None:
    wandb.init()

    # to reduce variance, do many runs, and optimise on the average
    results = []
    for i in range(wandb.config["n_runs"]):
        wandb.define_metric("target-classifier-accuracy", summary="mean")
        do_run()

    wandb.finish()


if __name__ == "__main__":
    main()
