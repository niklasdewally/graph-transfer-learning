# Data
This directory contains datasets (generated and empirical) used by
experiments.


# Generated Datasets

## Clustered

This dataset contains graphs generated using the same parameters, but with different
triangle densities.

Each file is a `networkx` compatible edge-list. Files are named according to the
following schema:

```
{graph type (powerlaw | poisson)}-{clustered | unclustered}-{number of nodes}-{i}.edgelist
```

For more details, see the `generate_clustered_dataset.py` script.


## `2023-07-clustered`

This folder contains clustered graphs generated for the August '23 set of experiments, as described in WORD DOCUMENT.

TODO (niklasdewally): Add this word document (once finalised) to `reports/`


File names obey the following schema:
```
{graph type}-{n_nodes}-{max-clique-size}-{i}-.gml
```

Pre-sampled negative triangles (for triangle detection experiments) are also in this directory. These are named as follows:

```
{graph-file-name}-negative-triangles.json
```


