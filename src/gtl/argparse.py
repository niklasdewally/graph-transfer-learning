"""
Standard commandline arguments for use in this project.
"""

from argparse import ArgumentParser

def add_wandb_options(parser: ArgumentParser) -> ArgumentParser:
    """
    Add flags to the given parser for controlling runs with wandb.

    Flags added are:

    --dev | -D         Set wandb to offine mode, allowing testing code without
                       uploading results to wandb.

    --disable-wandb    Disable wandb.
    """

    parser.add_argument("-D","--dev",
                        dest="mode",
                        action="store_const",
                        const="offline",
                        help=("run in development mode. This disables syncing"
                              " of wandb runs, but still shows run metrics in"
                              " the console"))

    parser.add_argument("--disable-wandb",
                        dest="mode",
                        action="store_const",
                        const="disabled",
                        help="disable wandb.")


    return parser
