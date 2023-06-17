# Data

This directory contains datasets (generated and emperical) used by my
experiments.


## Generated Datasets

### Clustered

This dataset contains graphs generated using the same parameters, but with different
triangle densities.

Each file is a `networkx` compatible edge-list. Files are named according to the
following schema:

```
{graph type (powerlaw | poisson)}-{clustered | unclustered}-{number of nodes}-{i}.edgelist
```

For more details, see the `generate_clustered_dataset.py` script.
