#!/usr/bin/env python3
"""Run Squidpy neighborhood enrichment as a colocalization baseline.

This script produces cell-type-pair neighborhood-enrichment z-scores from either
an AnnData .h5ad file or a coordinate/cell-type CSV such as data/atlas_allexp.csv.
It does not compare against ColocEM outputs; it only generates clean Squidpy
baseline tables for a later comparison step.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path


LOGGER = logging.getLogger("squidpy_enrichment")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute Squidpy neighborhood-enrichment z-scores."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input .h5ad file or coordinate/cell-type CSV file.",
    )
    parser.add_argument(
        "--cell-type-key",
        default="cell_type",
        help="Column in adata.obs or CSV containing cell-type labels.",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help="Key in adata.obsm for spatial coordinates; also used when building AnnData from CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default=Path("results/squidpy_enrichment"),
        type=Path,
        help="Directory where Squidpy enrichment CSV outputs will be written.",
    )
    parser.add_argument(
        "--n-perms",
        default=1000,
        type=int,
        help="Number of permutations for Squidpy neighborhood enrichment.",
    )
    parser.add_argument(
        "--radius",
        default=None,
        type=float,
        help="Radius passed to sq.gr.spatial_neighbors. Mutually exclusive with --n-neighs.",
    )
    parser.add_argument(
        "--n-neighs",
        default=None,
        type=int,
        help="Number of nearest neighbors passed to sq.gr.spatial_neighbors when --radius is not used.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Random seed passed to Squidpy neighborhood enrichment.",
    )
    parser.add_argument(
        "--n-jobs",
        default=1,
        type=int,
        help="Number of jobs passed to Squidpy neighborhood enrichment.",
    )
    parser.add_argument(
        "--max-estimated-edges",
        default=20_000_000,
        type=int,
        help=(
            "Safety limit for estimated directed graph edges when --radius is used. "
            "Increase only if you have enough memory."
        ),
    )
    parser.add_argument(
        "--radius-safety-sample",
        default=2000,
        type=int,
        help="Number of cells sampled to estimate graph density for --radius safety checks.",
    )
    parser.add_argument(
        "--force-radius",
        action="store_true",
        help="Skip the --radius graph-density safety check.",
    )
    parser.add_argument(
        "--x-col",
        default="x",
        help="CSV column containing x coordinates. Ignored for .h5ad input.",
    )
    parser.add_argument(
        "--y-col",
        default="y",
        help="CSV column containing y coordinates. Ignored for .h5ad input.",
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def read_csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"Input CSV is empty: {path}") from exc


def validate_input_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    if path.suffix.lower() not in {".h5ad", ".csv"}:
        raise ValueError(
            "Unsupported input format. Expected a .h5ad AnnData file or a .csv file."
        )


def load_csv_as_anndata(
    path: Path,
    *,
    cell_type_key: str,
    spatial_key: str,
    x_col: str,
    y_col: str,
):
    import numpy as np
    import pandas as pd
    from anndata import AnnData
    from scipy import sparse

    header = read_csv_header(path)
    required = [x_col, y_col, cell_type_key]
    missing = [col for col in required if col not in header]
    if missing:
        raise ValueError(
            f"Input CSV is missing required column(s): {missing}. "
            f"Available columns include: {header[:20]}"
        )

    LOGGER.info(
        "Reading CSV columns %s from %s. Gene-expression columns will not be loaded.",
        required,
        path,
    )
    df = pd.read_csv(path, usecols=required)
    if df.empty:
        raise ValueError(f"Input CSV contains no rows: {path}")

    coords = df[[x_col, y_col]].apply(pd.to_numeric, errors="coerce")
    bad_coord_rows = coords.isna().any(axis=1)
    if bad_coord_rows.any():
        n_bad = int(bad_coord_rows.sum())
        raise ValueError(
            f"CSV contains {n_bad} row(s) with missing or nonnumeric coordinates "
            f"in columns {x_col!r}/{y_col!r}."
        )

    labels = df[cell_type_key]
    if labels.isna().all():
        raise ValueError(f"CSV cell-type column {cell_type_key!r} is entirely missing.")

    keep = labels.notna()
    if (~keep).any():
        LOGGER.warning(
            "Dropping %d row(s) with missing cell-type labels in %r.",
            int((~keep).sum()),
            cell_type_key,
        )
        labels = labels.loc[keep]
        coords = coords.loc[keep]

    coords_array = coords.to_numpy(dtype=float)
    if coords_array.ndim != 2 or coords_array.shape[1] != 2:
        raise ValueError(
            f"CSV spatial coordinates must form an n x 2 matrix; got shape {coords_array.shape}."
        )
    if not np.isfinite(coords_array).all():
        raise ValueError("CSV spatial coordinates contain non-finite values.")

    obs = pd.DataFrame(index=[f"cell_{i}" for i in range(len(labels))])
    obs[cell_type_key] = labels.astype(str).to_numpy()
    obs[cell_type_key] = pd.Categorical(obs[cell_type_key])

    adata = AnnData(X=sparse.csr_matrix((len(obs), 0)), obs=obs)
    adata.obsm[spatial_key] = coords_array
    LOGGER.info(
        "Constructed AnnData from CSV with %d cells and %d cell-type categories.",
        adata.n_obs,
        len(adata.obs[cell_type_key].cat.categories),
    )
    return adata


def load_h5ad(
    path: Path,
    *,
    cell_type_key: str,
    spatial_key: str,
):
    import numpy as np
    import pandas as pd
    import scanpy as sc

    LOGGER.info("Reading AnnData file: %s", path)
    adata = sc.read_h5ad(path)

    if spatial_key not in adata.obsm:
        raise ValueError(
            f"AnnData is missing spatial coordinates in adata.obsm[{spatial_key!r}]. "
            f"Available obsm keys: {list(adata.obsm.keys())}"
        )
    if cell_type_key not in adata.obs:
        raise ValueError(
            f"AnnData is missing cell-type labels in adata.obs[{cell_type_key!r}]. "
            f"Available obs columns include: {list(adata.obs.columns[:20])}"
        )

    coords = np.asarray(adata.obsm[spatial_key])
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(
            f"adata.obsm[{spatial_key!r}] must be a 2D coordinate matrix with at least "
            f"two columns; got shape {coords.shape}."
        )
    if not np.isfinite(coords[:, :2]).all():
        raise ValueError(f"adata.obsm[{spatial_key!r}] contains non-finite coordinates.")

    labels = adata.obs[cell_type_key]
    if labels.isna().all():
        raise ValueError(f"AnnData obs column {cell_type_key!r} is entirely missing.")
    keep = labels.notna().to_numpy()
    if (~keep).any():
        LOGGER.warning(
            "Dropping %d cell(s) with missing cell-type labels in %r.",
            int((~keep).sum()),
            cell_type_key,
        )
        adata = adata[keep].copy()

    adata.obs[cell_type_key] = pd.Categorical(adata.obs[cell_type_key].astype(str))
    LOGGER.info(
        "Loaded AnnData with %d cells and %d cell-type categories.",
        adata.n_obs,
        len(adata.obs[cell_type_key].cat.categories),
    )
    return adata


def load_input(args: argparse.Namespace):
    validate_input_path(args.input)
    suffix = args.input.suffix.lower()
    if suffix == ".csv":
        return load_csv_as_anndata(
            args.input,
            cell_type_key=args.cell_type_key,
            spatial_key=args.spatial_key,
            x_col=args.x_col,
            y_col=args.y_col,
        )
    return load_h5ad(
        args.input,
        cell_type_key=args.cell_type_key,
        spatial_key=args.spatial_key,
    )


def run_squidpy_enrichment(adata, args: argparse.Namespace):
    import numpy as np
    import squidpy as sq

    np.random.seed(args.seed)

    neighbor_kwargs = {
        "spatial_key": args.spatial_key,
        "coord_type": "generic",
    }
    if args.radius is not None:
        neighbor_kwargs["radius"] = args.radius
    elif args.n_neighs is not None:
        neighbor_kwargs["n_neighs"] = args.n_neighs

    if args.radius is not None and not args.force_radius:
        check_radius_graph_size(adata, args)

    LOGGER.info("Building Squidpy spatial neighbor graph with %s.", neighbor_kwargs)
    sq.gr.spatial_neighbors(adata, **neighbor_kwargs)

    LOGGER.info(
        "Running Squidpy neighborhood enrichment with n_perms=%d and seed=%d.",
        args.n_perms,
        args.seed,
    )
    sq.gr.nhood_enrichment(
        adata,
        cluster_key=args.cell_type_key,
        n_perms=args.n_perms,
        seed=args.seed,
        n_jobs=args.n_jobs,
    )

    result_key = f"{args.cell_type_key}_nhood_enrichment"
    if result_key not in adata.uns:
        raise RuntimeError(
            f"Squidpy did not write expected result key adata.uns[{result_key!r}]. "
            f"Available uns keys: {list(adata.uns.keys())}"
        )
    if "zscore" not in adata.uns[result_key]:
        raise RuntimeError(
            f"Squidpy result adata.uns[{result_key!r}] does not contain a 'zscore' matrix."
        )
    return np.asarray(adata.uns[result_key]["zscore"])


def check_radius_graph_size(adata, args: argparse.Namespace) -> None:
    import numpy as np
    from scipy.spatial import cKDTree

    coords = np.asarray(adata.obsm[args.spatial_key])[:, :2]
    n_cells = coords.shape[0]
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    ranges = maxs - mins
    diagonal = float(np.linalg.norm(ranges))

    LOGGER.info(
        "Spatial coordinate range: x=[%.4g, %.4g], y=[%.4g, %.4g], diagonal=%.4g.",
        mins[0],
        maxs[0],
        mins[1],
        maxs[1],
        diagonal,
    )

    if args.radius >= diagonal:
        estimated_edges = n_cells * max(n_cells - 1, 0)
        raise ValueError(
            f"--radius {args.radius:g} is larger than or equal to the tissue bounding-box "
            f"diagonal ({diagonal:.4g}). This would create an almost complete graph "
            f"for {n_cells} cells (~{estimated_edges:,} directed edges) and can exhaust "
            "memory. For atlas_allexp.csv, use --n-neighs 6 or choose a radius in the "
            "same coordinate scale as x/y. Use --force-radius only if this is intentional."
        )

    sample_size = min(args.radius_safety_sample, n_cells)
    rng = np.random.default_rng(args.seed)
    sample_idx = rng.choice(n_cells, size=sample_size, replace=False)
    tree = cKDTree(coords)
    counts = tree.query_ball_point(
        coords[sample_idx],
        r=args.radius,
        return_length=True,
    )
    mean_neighbors = float(np.mean(np.maximum(counts - 1, 0)))
    max_neighbors = int(np.max(np.maximum(counts - 1, 0)))
    estimated_edges = int(round(mean_neighbors * n_cells))

    LOGGER.info(
        "Radius safety estimate: mean neighbors/cell=%.1f, max sampled neighbors=%d, "
        "estimated directed edges=%s.",
        mean_neighbors,
        max_neighbors,
        f"{estimated_edges:,}",
    )

    if estimated_edges > args.max_estimated_edges:
        raise ValueError(
            f"--radius {args.radius:g} is estimated to create ~{estimated_edges:,} "
            f"directed edges, above --max-estimated-edges={args.max_estimated_edges:,}. "
            "This can exhaust memory during Squidpy graph construction. Use --n-neighs "
            "for a bounded kNN graph, reduce --radius, raise --max-estimated-edges, or "
            "pass --force-radius if this dense graph is intentional."
        )


def save_outputs(adata, zscore, args: argparse.Namespace) -> None:
    import pandas as pd

    categories = list(adata.obs[args.cell_type_key].cat.categories)
    if zscore.shape != (len(categories), len(categories)):
        raise RuntimeError(
            "Z-score matrix shape does not match number of cell-type categories: "
            f"zscore={zscore.shape}, categories={len(categories)}."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = args.output_dir / "squidpy_nhood_enrichment_zscore_matrix.csv"
    long_path = args.output_dir / "squidpy_nhood_enrichment_zscores_long.csv"

    matrix_df = pd.DataFrame(zscore, index=categories, columns=categories)
    matrix_df.index.name = "cell_type_1"
    matrix_df.to_csv(matrix_path)

    long_df = (
        matrix_df.rename_axis(index="cell_type_1", columns="cell_type_2")
        .stack(dropna=False)
        .rename("squidpy_zscore")
        .reset_index()
    )
    long_df.to_csv(long_path, index=False)

    LOGGER.info("Saved square z-score matrix: %s", matrix_path)
    LOGGER.info("Saved long-format z-score table: %s", long_path)


def validate_args(args: argparse.Namespace) -> None:
    if args.radius is not None and args.n_neighs is not None:
        raise ValueError("--radius and --n-neighs are mutually exclusive.")
    if args.n_perms < 1:
        raise ValueError("--n-perms must be at least 1.")
    if args.radius is not None and args.radius <= 0:
        raise ValueError("--radius must be positive when provided.")
    if args.n_neighs is not None and args.n_neighs < 1:
        raise ValueError("--n-neighs must be at least 1 when provided.")
    if args.n_jobs < 1:
        raise ValueError("--n-jobs must be at least 1.")
    if args.max_estimated_edges < 1:
        raise ValueError("--max-estimated-edges must be at least 1.")
    if args.radius_safety_sample < 1:
        raise ValueError("--radius-safety-sample must be at least 1.")


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
        adata = load_input(args)
        zscore = run_squidpy_enrichment(adata, args)
        save_outputs(adata, zscore, args)
        LOGGER.info("Done.")
    except Exception as exc:
        LOGGER.exception("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
