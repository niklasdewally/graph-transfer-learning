"""
Shared functionality for command line programs and experiment scripts.


Includes:
    * standard commandline arguments for experiments.
    * pretty printing functionality.
"""

from argparse import ArgumentParser
import argparse


def standard_generator_parser() -> ArgumentParser:
    """
    Return a parser that contains the --overwrite, --dry-run, and --verbose
    flags for use in generator scripts.
    """
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def add_wandb_options(parser: ArgumentParser) -> ArgumentParser:
    """
    Add flags to the given parser for controlling runs with wandb.

    Flags added are:

    --dev | -D         Set wandb to offine mode, allowing testing code without
                       uploading results to wandb.

    --disable-wandb    Disable wandb.
    """

    parser.add_argument(
        "-D",
        "--dev",
        dest="mode",
        action="store_const",
        const="offline",
        help=(
            "run in development mode. This disables syncing"
            " of wandb runs, but still shows run metrics in"
            " the console"
        ),
    )

    parser.add_argument(
        "--disable-wandb",
        dest="mode",
        action="store_const",
        const="disabled",
        help="disable wandb.",
    )

    return parser


def print_title(text: str) -> None:
    """
    Print a title.

    Title text
    ==========

    """

    print(f"{text}\n{''.join(['=' for _ in text])}")
