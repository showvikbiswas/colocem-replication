# ColocEM Replication Package

This repository contains the implementation and replication materials for **ColocEM**, a method for modeling cell-cell communication-sensitive gene expression from spatial transcriptomics data.

ColocEM is designed for spatial datasets where each cell has:

- spatial coordinates,
- an assigned cell type or class,
- genome-wide or broad gene expression measurements.

The core idea is that genes affected by cell-cell communication are determined by the receiver cell's local spatial micro-environment and by ligand activity from neighboring sender cells, along with the receiver cell's own receptor expression. ColocEM combines these signals into an interpretable predictive model.

## Method Overview

ColocEM defines cellular micro-environments using spatial colocalization between cell-type pairs. Instead of treating a neighborhood as a fixed-radius binary graph, ColocEM estimates how strongly pairs of cell types co-occur across tissue space.

The main workflow is:

1. **Cell-type density estimation**
   A Gaussian kernel density estimate is fit for each cell type using spatial coordinates.

2. **Weighted colocalization scoring**
   The tissue is scanned with overlapping sliding windows. Within each window, ColocEM evaluates the cell-type KDEs on a grid and computes a weighted Pearson correlation coefficient for every cell-type pair. Weighting reduces the influence of regions where both cell types have very low density.

3. **Highly colocalized island detection**
   Windows with pairwise colocalization above a threshold are treated as graph nodes. Adjacent high-correlation windows are grouped into connected components, called **islands**. These islands represent spatial micro-environments where two cell types are strongly colocalized.

4. **Per-cell niche features**
   For each receiver cell, ColocEM computes island coverage features: the fraction of covering windows in which the receiver's cell type is colocalized with each potential sender type.

5. **Ligand exposure**
   For each receiver cell, ligand exposure is computed from neighboring sender cells inside relevant colocalized islands. The implementation supports mean exposure and Gaussian distance-weighted exposure.

6. **Expression prediction**
   For each receiver cell type, ColocEM trains XGBoost regressors to predict non-receptor, non-ligand target gene expression using:

   - receiver receptor expression,
   - ligand exposure,
   - island coverage features.

7. **Downstream analysis**
   The implementation reports per-gene `R2`, permutation-based p-values, ligand/receptor perturbation signatures, SHAP feature importance, and responsive-gene tables.

## Repository Structure

```text
.
├── Manuscript.pdf
├── README.md
├── requirements.txt
├── analysis/
│   ├── colocalization_breakdown.ipynb
│   └── r2-comparison.py
├── data/
│   ├── mouse_850_lr_pairs_cpdb_interactions.csv
│   └── r2_summary.csv
├── pipeline-notebooks/
│   └── atlas-pipeline.ipynb
└── scripts/
    ├── ncem_results.py
    └── pipeline.py
```

## Environment Setup

Create and activate a Python environment, then install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The main dependencies are:

- `numpy`, `pandas`, and `scipy` for data handling, KDEs, and island detection,
- `scikit-learn` for spatial grouping and cross-validation utilities,
- `xgboost` for per-gene expression models,
- `shap` for feature-importance analysis,
- `matplotlib` for summary plots,
- `joblib` for model bundle export,
- `tqdm` for progress bars.

## Input Data Format

The main pipeline expects a cell-by-gene CSV file. By default, the file should contain:

- `x`: cell x-coordinate,
- `y`: cell y-coordinate,
- `class`: receiver/sender cell type label,
- optional `cell_label`,
- gene expression columns.

Example:

```text
x,y,class,cell_label,GeneA,GeneB,GeneC,...
12.4,8.1,Excitatory,c1,0.0,1.3,2.1,...
13.2,8.5,Inhibitory,c2,0.4,0.0,1.7,...
```

If your metadata column names differ, pass them through command-line options such as `--x-col`, `--y-col`, `--class-col`, and `--meta-cols`.

The default ligand-receptor file is:

```text
data/atlas_allexp.csv
```

It is expected to contain ligand and receptor gene-symbol columns. By default, these are:

- `ligand_genesymbol`,
- `target_genesymbol`.

Use `--ligand-col` and `--receptor-col` if your ligand-receptor table uses different names.

## Quick Start: Running the Pipeline

Run the full ColocEM implementation with:

```bash
python scripts/pipeline.py path/to/expression.csv
```

By default, results are written to:

```text
results/
```

You can choose a different output directory:

```bash
python scripts/pipeline.py path/to/expression.csv --results-dir results/atlas_run
```

Useful options include:

```bash
python scripts/pipeline.py path/to/expression.csv \
  --lr-pairs data/mouse_850_lr_pairs_cpdb_interactions.csv \
  --win-size 2.0 \
  --grid-n 25 \
  --kde-bw 0.2 \
  --island-r-threshold 0.7 \
  --island-min-windows 4 \
  --coverage-theta 0.5 \
  --exposure-mode kde \
  --n-groups 8 \
  --target-limit 300
```

For a faster exploratory run, reduce the number of target genes, permutations, or XGBoost search settings:

```bash
python scripts/pipeline.py path/to/expression.csv \
  --target-limit 50 \
  --n-permutations 0 \
  --max-depths 4 \
  --learning-rates 0.1 \
  --subsample 0.8 \
  --colsample 0.8 \
  --reg-l2 5 \
  --reg-l1 0 \
  --skip-shap
```

## Pipeline Outputs

The script writes intermediate and downstream outputs under the selected results directory.

Important files include:

- `pairwise_weighted_pcc_map.csv`: weighted PCC values for cell-type pairs across sliding windows.
- `island_index.csv`: detected colocalized islands and summary statistics.
- `binary_colocalization.csv`: per-cell binary colocalization encoding.
- `cell_coverage.csv`: per-cell island coverage features.
- `lr_pairs_kept.csv`: ligand-receptor pairs retained after matching genes to the expression table.
- `lr_feature_report.json`: ligand-receptor matching summary.
- `features/x_receptors.csv`: receiver receptor expression matrix.
- `features/x_ligand_exposure.csv`: ligand exposure matrix.
- `features/x_coverage.csv`: island coverage feature matrix.
- `features/target_genes.csv`: target genes modeled by XGBoost.
- `r2_summary.csv`: per-receiver, per-gene XGBoost performance.
- `split_info.json`: spatial split and feature metadata.
- `downstream/r2_ecdf.png`: ECDF plot of per-gene `R2` values.
- `downstream/p_values.csv`: permutation-based p-values, when enabled.
- `downstream/responsive_genes/`: per-receiver responsive-gene tables.
- `downstream/shap/`: SHAP feature rankings, when enabled.
- `downstream/perturbations/`: ligand blockade and receptor knockout signatures.
- `models/`: trained XGBoost model files, when model saving is enabled.

## Key Implementation Files

### `scripts/pipeline.py`

This is the main command-line implementation of ColocEM. It runs the active pipeline path used for the atlas analysis:

- weighted PCC only,
- connected-component island formation,
- island coverage feature construction,
- ligand exposure computation,
- XGBoost model training,
- per-gene `R2` evaluation,
- permutation p-value estimation,
- responsive-gene identification,
- SHAP feature ranking,
- ligand and receptor perturbation analysis,
- export of results and trained models.

The script intentionally does **not** use the ElasticNet experiment from the notebook and does **not** use explicit ligand-receptor product features in the final XGBoost design matrix. The XGBoost model uses receptor expression, ligand exposure, and coverage features directly.

### `pipeline-notebooks/atlas-pipeline.ipynb`

This notebook is the exploratory implementation of the atlas pipeline. It contains the development version of the workflow, including:

- unweighted and weighted PCC experiments,
- island formation,
- coverage encoding,
- ligand-receptor feature preparation,
- ligand exposure experiments,
- ElasticNet exploration,
- XGBoost training,
- p-value analysis,
- SHAP analysis,
- perturbation analysis,
- model/result save-load cells.

Not every notebook section is part of the final streamlined pipeline. The current command-line script follows the weighted PCC and XGBoost path.

### `scripts/ncem_results.py`

This script supports comparison against NCEM-style outputs, especially per-gene `R2` extraction from NCEM-generated predictions. It is separate from the ColocEM pipeline itself.

### `analysis/`

The `analysis/` directory contains downstream analysis utilities and notebooks, including colocalization breakdown and `R2` comparison code.

### `data/`

The `data/` directory contains small repository data assets such as ligand-receptor pairs and existing `R2` summaries. Full spatial transcriptomics datasets may need to be downloaded separately depending on the analysis being reproduced.

## Reproducibility Notes

ColocEM uses spatially aware train/test splitting. Cell coordinates are clustered into spatial groups with KMeans, and held-out spatial groups are used for test evaluation. This helps reduce leakage between nearby cells.

Default model settings are chosen to follow the notebook implementation, but many hyperparameters can be controlled through CLI flags:

- sliding-window and KDE parameters,
- island threshold and minimum island size,
- ligand exposure mode,
- target filtering thresholds,
- number of spatial groups,
- XGBoost search space,
- number of permutation runs,
- SHAP and model-saving behavior.

For final manuscript-scale analysis, use the full dataset and increase permutation runs as needed. For debugging or exploratory runs, reduce `--target-limit`, set `--n-permutations 0`, and use `--skip-shap`.

## Citation

If you use this repository, please cite the accompanying ColocEM manuscript:

**ColocEM: modeling cell-cell communication sensitive gene expressions based on colocalization techniques from spatial transcriptomics data.**
