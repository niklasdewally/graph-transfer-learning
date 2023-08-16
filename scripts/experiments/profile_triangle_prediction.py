"""
Profile the triangle prediction experiment on 10,000 sized graphs.

Creates a file called trace.json.
"""
# pyre-fixme[21]:
import wandb

import sys

from aug_triangle_prediction import default_config, do_run, HYPERPARAMS_DIR
import torch.profiler
import gtl


def main() -> int:
    # create dummy run just to have config dict
    wandb.init(
        project="August 2023 01 - Triangle Detection",
        mode="disabled",
        entity="sta-graph-transfer-learning",
        config=default_config,
        name="DEV",
    )
    wandb.config["source_size"] = "10000"
    wandb.config["target_size"] = "10000"
    wandb.config["model"] = "triangle"

    model_config = gtl.load_model_config(HYPERPARAMS_DIR, wandb.config["model"])
    wandb.config.update(model_config)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        profile_memory=True,
        with_flops=True,
        record_shapes=True,
    ) as p:
        do_run()

    p.export_chrome_trace("trace.json")
    wandb.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
