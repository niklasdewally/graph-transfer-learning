import tomllib
from .typing import PathLike
from pathlib import Path
from collections.abc import MutableMapping


def load_model_config(dir: PathLike, model: str) -> MutableMapping:
    """
    Load model hyperparameters from a toml file
    """

    with open(Path(dir) / f"{model}.toml", "rb") as f:
        config = tomllib.load(f)
    return config
