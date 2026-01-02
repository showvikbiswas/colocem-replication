# colocem-replication

This repository contains the replication package for ColocEM. The repository is structured as follows:

- `data/` contains one sample timepoint from the ARTISTA dataset (see Methods) for full dataset, and the gene expressions for the Mouse Brain Atlas dataset. 
- `analysis/` contains some notebooks to reproduce our downstream analyses.
- `pipeline-noteboks/` contains the notebook to run the ColocEM pipeline notebooks from start to end using the Mouse Brain Atlas dataset.
- `scripts/` contains the required code to run Linear NCEM on the ARTISTA datapoint provided. You can download further datasets and modify this file accordingly.
- `results/xgb/` contains the values produced by ColocEM using a XGBoost model. We make direct comparisons on the ARTISTA dataset with these values.