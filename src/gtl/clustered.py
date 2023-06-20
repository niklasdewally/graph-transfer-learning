"""
Utility functions for the clustered graph synthetic data-set
"""


def get_filename(
    graph_type: str,
    is_clustered: bool,
    size: int,
    i: int,
):
    """
    Return the expected filename for the clustered graph with the given
    parameters.
    """

    clustered_str = "clustered" if is_clustered else "unclustered"
    return f"{graph_type}-{clustered_str}-{size}-{i}.edgelist"
