"""
Profile the triangle prediction experiment on 10,000 sized graphs.

Creates a file called trace.json.
"""
# pyre-fixme[21]:
import wandb

import sys

from aug_triangle_prediction import default_config, do_run, HYPERPARAMS_DIR
import gtl


def main() -> int:
    # create dummy run just to have config dict
    wandb.init(
        project="August 2023 01 - Triangle Detection",
        entity="sta-graph-transfer-learning",
        config=default_config,
        tags=["debug"],
    )
    wandb.config["source_size"] = "10000"
    wandb.config["target_size"] = "10000"
    wandb.config["model"] = "graphsage-mean"

    model_config = gtl.load_model_config(HYPERPARAMS_DIR, wandb.config["model"])
    wandb.config.update(model_config)
    do_run()

    wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
