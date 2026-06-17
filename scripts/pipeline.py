#!/usr/bin/env python3
"""Run the ColocEM atlas-style pipeline from a CSV expression table.

The script implements the active notebook path:
weighted PCC -> colocalized islands -> receptor / ligand-exposure / coverage
features -> per-receiver XGBoost models -> downstream summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from scipy import ndimage
from scipy.stats import gaussian_kde
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from xgboost import XGBRegressor


@dataclass
class PipelineConfig:
    csv_path: Path
    lr_pairs_path: Path
    results_dir: Path
    x_col: str = "x"
    y_col: str = "y"
    class_col: str = "class"
    cell_label_col: str = "cell_label"
    ligand_col: str = "ligand_genesymbol"
    receptor_col: str = "target_genesymbol"
    meta_cols: tuple[str, ...] = ("x", "y", "class", "cell_label")
    gene_start: int | None = None
    gene_end: int | None = None
    win_size: float = 2.0
    grid_n: int = 25
    kde_bw: float | str = 0.2
    min_kde_points: int = 5
    weight_mode: str = "sum"
    pcc_eps: float = 1e-12
    island_r_threshold: float = 0.7
    island_min_windows: int = 4
    coverage_theta: float = 0.5
    coverage_min_support: int = 3
    exposure_mode: str = "kde"
    exposure_sigma: float | None = None
    min_detect_frac: float = 0.01
    min_var_quantile: float = 0.05
    drop_ligands: bool = True
    drop_technicals: bool = True
    n_groups: int = 8
    test_fraction: float = 0.2
    n_splits: int = 5
    target_limit: int | None = 300
    ignore_zero_cov: bool = True
    use_sample_weights: bool = True
    seed: int = 42
    n_estimators: int = 2000
    early_stop: int = 100
    tree_method: str = "hist"
    n_jobs: int = 0
    max_depths: tuple[int, ...] = (4, 6)
    learning_rates: tuple[float, ...] = (0.03, 0.1)
    subsample: tuple[float, ...] = (0.8, 1.0)
    colsample: tuple[float, ...] = (0.8, 1.0)
    reg_l2: tuple[float, ...] = (1.0, 5.0, 10.0)
    reg_l1: tuple[float, ...] = (0.0, 1.0)
    n_permutations: int = 5
    top_genes_downstream: int = 50
    shap_sample: int = 5000
    skip_shap: bool = False
    save_models: bool = True


def log(message: str) -> None:
    print(f"[ColocEM] {message}", flush=True)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def parse_number_list(value: str, cast: type) -> tuple:
    return tuple(cast(v.strip()) for v in value.split(",") if v.strip())


def load_expression_table(config: PipelineConfig) -> pd.DataFrame:
    df = pd.read_csv(config.csv_path)
    required = [config.x_col, config.y_col, config.class_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")
    return df.reset_index(drop=True)


def build_global_kdes(
    df: pd.DataFrame,
    config: PipelineConfig,
) -> dict[str, gaussian_kde]:
    kdes = {}
    for ctype, sub in df.groupby(config.class_col):
        pts = sub[[config.x_col, config.y_col]].to_numpy(float)
        if pts.shape[0] >= config.min_kde_points:
            kdes[str(ctype)] = gaussian_kde(pts.T, bw_method=config.kde_bw)
    return kdes


def sliding_windows(xmin: float, xmax: float, ymin: float, ymax: float, size: float):
    step = size / 2.0

    def starts(lo: float, hi: float) -> list[float]:
        s = []
        cur = lo
        while cur + size <= hi + 1e-9:
            s.append(cur)
            cur += step
        if not s or s[-1] + size < hi:
            s.append(max(lo, hi - size))
        return sorted(set(float(v) for v in s))

    for x0 in starts(xmin, xmax):
        for y0 in starts(ymin, ymax):
            yield (x0, x0 + size, y0, y0 + size)


def grid_cell_centers(x0: float, x1: float, y0: float, y1: float, n: int):
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n
    xs = x0 + hx * (np.arange(n) + 0.5)
    ys = y0 + hy * (np.arange(n) + 0.5)
    gx, gy = np.meshgrid(xs, ys, indexing="xy")
    return gx, gy, hx, hy


def windowwise_normalize(z: np.ndarray, hx: float, hy: float) -> np.ndarray:
    mass = z.sum() * hx * hy
    return z / mass if mass > 0 else z


def weighted_pearson(a: np.ndarray, b: np.ndarray, mode: str, eps: float) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    w = a * b if mode == "prod" else a + b
    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= eps:
        return np.nan
    w = w / w_sum
    mu_a = (w * a).sum()
    mu_b = (w * b).sum()
    da = a - mu_a
    db = b - mu_b
    var_a = (w * da * da).sum()
    var_b = (w * db * db).sum()
    if var_a <= eps or var_b <= eps:
        return np.nan
    cov_ab = (w * da * db).sum()
    return float(cov_ab / np.sqrt(var_a * var_b))


def compute_pairwise_weighted_pcc(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    xmin, xmax = df[config.x_col].min(), df[config.x_col].max()
    ymin, ymax = df[config.y_col].min(), df[config.y_col].max()

    kdes = build_global_kdes(df, config)
    classes = sorted(kdes.keys())
    pairs = list(combinations(classes, 2))
    records = []

    windows = list(sliding_windows(xmin, xmax, ymin, ymax, config.win_size))
    for x0, x1, y0, y1 in tqdm(windows, desc="Weighted PCC windows"):
        gx, gy, hx, hy = grid_cell_centers(x0, x1, y0, y1, config.grid_n)
        xy = np.vstack([gx.ravel(), gy.ravel()])

        for a_type, b_type in pairs:
            za = kdes[a_type](xy).reshape(gx.shape)
            zb = kdes[b_type](xy).reshape(gx.shape)
            za = windowwise_normalize(za, hx, hy)
            zb = windowwise_normalize(zb, hx, hy)
            r_w = weighted_pearson(za, zb, mode=config.weight_mode, eps=config.pcc_eps)
            records.append(
                {
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                    "class_A": a_type,
                    "class_B": b_type,
                    "weighted_pearson_r": r_w,
                    "grid_n": config.grid_n,
                    "win_size": config.win_size,
                    "weight_mode": config.weight_mode,
                }
            )

    return pd.DataFrame.from_records(records)


def fisher_z(r: Any, eps: float = 1e-12):
    arr = np.asarray(r, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    arr = np.clip(arr, -1 + eps, 1 - eps)
    z = np.arctanh(arr)
    return np.where(np.isinf(z), np.nan, z)


def build_pair_grids(results_df: pd.DataFrame, a_type: str, b_type: str):
    sub = results_df[
        (results_df["class_A"] == a_type) & (results_df["class_B"] == b_type)
    ].copy()
    if sub.empty:
        return None
    sub = sub.drop_duplicates(subset=["x0", "y0", "x1", "y1", "class_A", "class_B"])
    xs = np.array(sorted(sub["x0"].unique()))
    ys = np.array(sorted(sub["y0"].unique()))
    ix = {x0: i for i, x0 in enumerate(xs)}
    iy = {y0: i for i, y0 in enumerate(ys)}
    shape = (len(ys), len(xs))
    grids = {
        "xs": xs,
        "ys": ys,
        "r_grid": np.full(shape, np.nan),
        "z_grid": np.full(shape, np.nan),
        "x0_grid": np.full(shape, np.nan),
        "x1_grid": np.full(shape, np.nan),
        "y0_grid": np.full(shape, np.nan),
        "y1_grid": np.full(shape, np.nan),
    }

    for _, row in sub.iterrows():
        i = iy[row["y0"]]
        j = ix[row["x0"]]
        r = row["weighted_pearson_r"]
        grids["r_grid"][i, j] = r
        grids["z_grid"][i, j] = fisher_z(r)
        grids["x0_grid"][i, j] = row["x0"]
        grids["x1_grid"][i, j] = row["x1"]
        grids["y0_grid"][i, j] = row["y0"]
        grids["y1_grid"][i, j] = row["y1"]
    return grids


def find_islands_for_all_pairs(
    results_df: pd.DataFrame,
    r_threshold: float,
    min_windows: int,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    islands = []
    index_rows = []
    z_thr = float(fisher_z(r_threshold))
    pairs = (
        results_df[["class_A", "class_B"]]
        .drop_duplicates()
        .sort_values(["class_A", "class_B"])
        .itertuples(index=False, name=None)
    )

    for a_type, b_type in pairs:
        grids = build_pair_grids(results_df, a_type, b_type)
        if grids is None:
            continue
        r_grid = grids["r_grid"]
        z_grid = grids["z_grid"]
        valid = np.isfinite(r_grid) & (r_grid > r_threshold)
        if not np.any(valid):
            continue

        labels, n_labels = ndimage.label(valid, structure=np.ones((3, 3), dtype=int))
        for label in range(1, n_labels + 1):
            mask = labels == label
            size = int(mask.sum())
            if size < min_windows:
                continue

            r_vals = r_grid[mask]
            z_vals = z_grid[mask]
            x0_min = float(np.nanmin(grids["x0_grid"][mask]))
            x1_max = float(np.nanmax(grids["x1_grid"][mask]))
            y0_min = float(np.nanmin(grids["y0_grid"][mask]))
            y1_max = float(np.nanmax(grids["y1_grid"][mask]))
            member_rects = np.column_stack(
                [
                    grids["x0_grid"][mask],
                    grids["x1_grid"][mask],
                    grids["y0_grid"][mask],
                    grids["y1_grid"][mask],
                ]
            ).tolist()

            island = {
                "pair": (a_type, b_type),
                "label": int(label),
                "n_windows": size,
                "median_r": float(np.nanmedian(r_vals)),
                "mean_r": float(np.nanmean(r_vals)),
                "max_r": float(np.nanmax(r_vals)),
                "median_z": float(np.nanmedian(z_vals)),
                "cluster_mass_z": float(np.nansum(z_vals - z_thr)),
                "bbox": (x0_min, x1_max, y0_min, y1_max),
                "window_rects": member_rects,
                "grid_shape": r_grid.shape,
                "grid_x0s": grids["xs"].tolist(),
                "grid_y0s": grids["ys"].tolist(),
            }
            islands.append(island)
            index_rows.append(
                {
                    "class_A": a_type,
                    "class_B": b_type,
                    "label": label,
                    "n_windows": size,
                    "bbox_x0": x0_min,
                    "bbox_x1": x1_max,
                    "bbox_y0": y0_min,
                    "bbox_y1": y1_max,
                    "cluster_mass_z": island["cluster_mass_z"],
                    "median_r": island["median_r"],
                }
            )

    island_index = pd.DataFrame(index_rows)
    if not island_index.empty:
        island_index = island_index.sort_values(
            ["class_A", "class_B", "cluster_mass_z", "n_windows"],
            ascending=[True, True, False, False],
        ).reset_index(drop=True)
    return islands, island_index


def island_windows_by_pair(islands: list[dict[str, Any]]):
    pair_to_windows = defaultdict(set)
    for island in islands:
        a_type, b_type = island["pair"]
        for x0, _, y0, _ in island["window_rects"]:
            pair_to_windows[(a_type, b_type)].add((float(x0), float(y0)))
    return pair_to_windows


def extract_grid_from_results(results_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    xs = np.array(sorted(results_df["x0"].unique()), dtype=float)
    ys = np.array(sorted(results_df["y0"].unique()), dtype=float)
    row = results_df.iloc[0]
    win_x = float(row["x1"] - row["x0"])
    win_y = float(row["y1"] - row["y0"])
    if not np.isclose(win_x, win_y):
        raise ValueError("Non-square windows are not supported.")
    return xs, ys, win_x


def covering_windows(x: float, y: float, xs: np.ndarray, ys: np.ndarray, win_size: float):
    x_mask = (xs <= x) & (x < xs + win_size)
    y_mask = (ys <= y) & (y < ys + win_size)
    xi = np.where(x_mask)[0]
    yi = np.where(y_mask)[0]
    return [(float(xs[j]), float(ys[i])) for i in yi for j in xi]


def lookup_pair_windows(pair_to_windows: dict, a_type: str, b_type: str):
    key = (a_type, b_type) if (a_type, b_type) in pair_to_windows else (b_type, a_type)
    return pair_to_windows.get(key, set())


def encode_cell_colocalization(
    df: pd.DataFrame,
    results_df: pd.DataFrame,
    islands: list[dict[str, Any]],
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    xs, ys, win_size = extract_grid_from_results(results_df)
    pair_to_windows = island_windows_by_pair(islands)
    cell_types = sorted(df[config.class_col].astype(str).unique())
    n = len(df)
    bin_data = {t: np.zeros(n, dtype=int) for t in cell_types}
    aux_rows = []

    all_cover = [
        covering_windows(float(row[config.x_col]), float(row[config.y_col]), xs, ys, win_size)
        for _, row in df[[config.x_col, config.y_col]].iterrows()
    ]

    for pos, row in df.iterrows():
        a_type = str(row[config.class_col])
        covered = all_cover[pos]
        support = len(covered)
        covered_set = set(covered)
        cov_map = {b_type: 0.0 for b_type in cell_types}

        if support >= config.coverage_min_support:
            for b_type in cell_types:
                if b_type == a_type:
                    continue
                windows = lookup_pair_windows(pair_to_windows, a_type, b_type)
                hit = len(covered_set & windows)
                coverage = hit / support if support else 0.0
                cov_map[b_type] = coverage
                if coverage >= config.coverage_theta:
                    bin_data[b_type][pos] = 1

        aux_row = {
            "cell_index": pos,
            config.x_col: float(row[config.x_col]),
            config.y_col: float(row[config.y_col]),
            "cell_type": a_type,
            "support_windows": support,
        }
        aux_row.update({f"coverage_{b_type}": float(cov_map[b_type]) for b_type in cell_types})
        aux_rows.append(aux_row)

    binary_mat = pd.DataFrame(bin_data, index=df.index)
    for a_type in cell_types:
        binary_mat.loc[df[config.class_col].astype(str) == a_type, a_type] = 0
    aux = pd.DataFrame(aux_rows).set_index("cell_index").loc[df.index]
    return binary_mat, aux


def prepare_lr_features(
    df: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    def upper_series(series: pd.Series) -> pd.Series:
        return series.astype(str).str.strip().str.upper()

    if config.ligand_col not in lr_pairs.columns or config.receptor_col not in lr_pairs.columns:
        raise ValueError(
            f"LR pairs CSV must contain {config.ligand_col!r} and {config.receptor_col!r}."
        )

    meta_cols = [c for c in config.meta_cols if c in df.columns]
    gene_cols = [c for c in df.columns if c not in meta_cols]
    gene_cols_upper = pd.Index([str(c).upper() for c in gene_cols])
    colmap = dict(zip(gene_cols_upper, gene_cols))
    genes_in_df_uc = set(gene_cols_upper)

    lr_uc = lr_pairs.copy()
    lr_uc["_LIG"] = upper_series(lr_pairs[config.ligand_col])
    lr_uc["_REC"] = upper_series(lr_pairs[config.receptor_col])
    keep_mask = lr_uc["_LIG"].isin(genes_in_df_uc) & lr_uc["_REC"].isin(genes_in_df_uc)
    lr_pairs_kept = lr_uc.loc[keep_mask].reset_index(drop=True)
    if lr_pairs_kept.empty:
        raise ValueError("No ligand-receptor pairs matched genes in the expression CSV.")

    uniq_lig_uc = list(dict.fromkeys(lr_pairs_kept["_LIG"]))
    uniq_rec_uc = list(dict.fromkeys(lr_pairs_kept["_REC"]))
    uniq_lig_cols = [colmap[g] for g in uniq_lig_uc]
    uniq_rec_cols = [colmap[g] for g in uniq_rec_uc]

    x_ligands = df.loc[:, uniq_lig_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    x_receptors = df.loc[:, uniq_rec_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    lig_idx_map = {g: i for i, g in enumerate(uniq_lig_cols)}
    rec_idx_map = {g: i for i, g in enumerate(uniq_rec_cols)}
    lr_pairs_kept["ligand_symbol_uc"] = lr_pairs_kept["_LIG"]
    lr_pairs_kept["receptor_symbol_uc"] = lr_pairs_kept["_REC"]
    lr_pairs_kept["ligand_symbol"] = lr_pairs_kept["_LIG"].map(colmap)
    lr_pairs_kept["receptor_symbol"] = lr_pairs_kept["_REC"].map(colmap)
    lr_pairs_kept["ligand_idx"] = lr_pairs_kept["ligand_symbol"].map(lig_idx_map)
    lr_pairs_kept["receptor_idx"] = lr_pairs_kept["receptor_symbol"].map(rec_idx_map)

    dropped_pairs = lr_uc.loc[~keep_mask, [config.ligand_col, config.receptor_col]]
    report = {
        "n_pairs_input": int(len(lr_pairs)),
        "n_pairs_kept": int(len(lr_pairs_kept)),
        "n_unique_ligands_kept": int(len(uniq_lig_cols)),
        "n_unique_receptors_kept": int(len(uniq_rec_cols)),
        "missing_ligands_from_expression": sorted(set(lr_uc["_LIG"]) - genes_in_df_uc),
        "missing_receptors_from_expression": sorted(set(lr_uc["_REC"]) - genes_in_df_uc),
        "n_dropped_pairs": int(len(dropped_pairs)),
    }
    lr_pairs_kept = lr_pairs_kept.drop(columns=["_LIG", "_REC"])
    return x_receptors, x_ligands, lr_pairs_kept, report


def preindex_window_cells(
    df: pd.DataFrame,
    xs: np.ndarray,
    ys: np.ndarray,
    win_size: float,
    config: PipelineConfig,
):
    x = df[config.x_col].to_numpy(float)
    y = df[config.y_col].to_numpy(float)
    classes = df[config.class_col].astype(str).to_numpy()
    win_index = {}
    for x0 in xs:
        x_mask = (x >= x0) & (x < x0 + win_size)
        for y0 in ys:
            y_mask = (y >= y0) & (y < y0 + win_size)
            idx = np.where(x_mask & y_mask)[0]
            if idx.size == 0:
                continue
            by_class = {}
            for ctype in np.unique(classes[idx]):
                by_class[ctype] = idx[classes[idx] == ctype]
            win_index[(float(x0), float(y0))] = by_class
    return win_index


def compute_ligand_exposure(
    df: pd.DataFrame,
    x_ligands: pd.DataFrame,
    results_df: pd.DataFrame,
    islands: list[dict[str, Any]],
    aux: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    xs, ys, win_size = extract_grid_from_results(results_df)
    pair_to_windows = island_windows_by_pair(islands)
    win_index = preindex_window_cells(df, xs, ys, win_size, config)
    coverage_cols = {
        c.replace("coverage_", ""): c for c in aux.columns if c.startswith("coverage_")
    }
    x_exposure = pd.DataFrame(0.0, index=df.index, columns=x_ligands.columns)
    coords = df[[config.x_col, config.y_col]].to_numpy(float)
    classes = df[config.class_col].astype(str).to_numpy()
    sigma = config.exposure_sigma
    if sigma is None:
        sigma = win_size / 3.0 if np.isfinite(win_size) and win_size > 0 else 1.0
    inv2sig2 = 1.0 / (2.0 * sigma**2)

    for i in tqdm(range(len(df)), desc="Ligand exposure"):
        a_type = classes[i]
        covered_i = covering_windows(coords[i, 0], coords[i, 1], xs, ys, win_size)
        support = len(covered_i)
        if support < config.coverage_min_support:
            continue
        eligible_b = [
            b_type
            for b_type, col in coverage_cols.items()
            if b_type != a_type and aux.iloc[i][col] >= config.coverage_theta
        ]
        if not eligible_b:
            continue

        nbr_idx = set()
        covered_set = set(covered_i)
        for b_type in eligible_b:
            island_windows = lookup_pair_windows(pair_to_windows, a_type, b_type) & covered_set
            for window in island_windows:
                by_class = win_index.get(window)
                if not by_class:
                    continue
                idx_b = by_class.get(b_type)
                if idx_b is not None and idx_b.size:
                    nbr_idx.update(idx_b.tolist())
        if not nbr_idx:
            continue

        nbr_idx_arr = np.fromiter(nbr_idx, dtype=int, count=len(nbr_idx))
        if config.exposure_mode == "mean":
            values = x_ligands.iloc[nbr_idx_arr, :].mean(axis=0).fillna(0.0).to_numpy()
        else:
            d2 = np.sum((coords[nbr_idx_arr] - coords[i]) ** 2, axis=1)
            weights = np.exp(-d2 * inv2sig2)
            w_sum = weights.sum()
            if w_sum <= 0:
                continue
            values = np.dot(weights / w_sum, x_ligands.iloc[nbr_idx_arr, :].to_numpy())
        x_exposure.iloc[i, :] = values
    return x_exposure


def get_gene_matrix(df: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    if config.gene_start is not None or config.gene_end is not None:
        cols = df.columns[slice(config.gene_start, config.gene_end)]
        expr = df.loc[:, cols]
    else:
        meta_cols = [c for c in config.meta_cols if c in df.columns]
        expr = df.drop(columns=meta_cols, errors="ignore")
    return expr.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def make_targets(
    df: pd.DataFrame,
    lr_pairs_kept: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    expr_all = get_gene_matrix(df, config)
    receptors = set(lr_pairs_kept["receptor_symbol"])
    ligands = set(lr_pairs_kept["ligand_symbol"])
    drop = set(expr_all.columns).intersection(receptors)
    if config.drop_ligands:
        drop |= set(expr_all.columns).intersection(ligands)

    det_frac = (expr_all > 0).mean(axis=0)
    var_g = expr_all.var(axis=0)
    drop |= set(det_frac[det_frac < config.min_detect_frac].index)
    drop |= set(var_g[var_g < var_g.quantile(config.min_var_quantile)].index)
    if config.drop_technicals:
        tech_re = re.compile(r"^(MT-|mt-|RPL|RPS|HBA|HBB)")
        drop |= {g for g in expr_all.columns if tech_re.match(str(g))}

    target_genes = [g for g in expr_all.columns if g not in drop]
    if not target_genes:
        raise ValueError("No target genes remained after filtering.")
    return expr_all.loc[:, target_genes].copy()


def make_coverage_features(aux: pd.DataFrame) -> pd.DataFrame:
    coverage_cols = [c for c in aux.columns if c.startswith("coverage_")]
    if not coverage_cols:
        raise ValueError("No coverage_* columns were created.")
    x_cov = aux[coverage_cols].copy()
    x_cov.columns = [f"cov::{c.replace('coverage_', '')}" for c in coverage_cols]
    return x_cov


def make_spatial_groups(xy: np.ndarray, n_groups: int, seed: int) -> np.ndarray:
    k = int(min(max(n_groups, 3), len(xy)))
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(xy)


def split_dev_test_by_groups(
    groups: np.ndarray,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    n_test = max(1, int(round(test_fraction * len(unique_groups))))
    test_groups = rng.choice(unique_groups, size=n_test, replace=False)
    is_test = np.isin(groups, test_groups)
    return ~is_test, is_test, test_groups


def r2_weighted(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        return float(r2_score(y_true, y_pred))
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)
    if weights.size == 0:
        return np.nan
    weights = weights / (weights.sum() + 1e-12)
    y_bar = np.sum(weights * y_true)
    sse = np.sum(weights * (y_true - y_pred) ** 2)
    sst = np.sum(weights * (y_true - y_bar) ** 2)
    return float(1.0 - sse / (sst + 1e-12))


def xgb_param_grid(config: PipelineConfig):
    return list(
        product(
            config.max_depths,
            config.learning_rates,
            config.subsample,
            config.colsample,
            config.reg_l2,
            config.reg_l1,
        )
    )


def make_xgb(params: dict[str, Any], config: PipelineConfig, n_estimators: int | None = None):
    return XGBRegressor(
        **params,
        n_estimators=n_estimators or config.n_estimators,
        objective="reg:squarederror",
        tree_method=config.tree_method,
        random_state=config.seed,
        n_jobs=config.n_jobs,
        eval_metric="rmse",
        early_stopping_rounds=config.early_stop,
    )


def train_xgb_for_receiver(
    receiver_type: str,
    df: pd.DataFrame,
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    aux: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[pd.DataFrame, dict[str, XGBRegressor], dict[str, Any]]:
    idx_all = df.index[df[config.class_col].astype(str) == receiver_type]
    x_block = pd.concat(
        [x_receptors.loc[idx_all], x_exposure.loc[idx_all], x_cov.loc[idx_all]], axis=1
    )
    y_block = y_targets.loc[idx_all]
    xy = df.loc[idx_all, [config.x_col, config.y_col]].to_numpy(float)
    weights = (
        aux.loc[idx_all, "support_windows"].clip(lower=1).to_numpy()
        if config.use_sample_weights
        else None
    )

    if config.ignore_zero_cov:
        keep = (x_cov.loc[idx_all].sum(axis=1) > 0).to_numpy()
        x_block = x_block.loc[keep]
        y_block = y_block.loc[keep]
        xy = xy[keep]
        if weights is not None:
            weights = weights[keep]

    if len(x_block) < 5 or y_block.shape[1] == 0:
        return pd.DataFrame(), {}, {"receiver_type": receiver_type, "skipped": True}

    if config.target_limit is not None:
        top_genes = y_block.var(axis=0).sort_values(ascending=False).index[: config.target_limit]
        y_block = y_block.loc[:, top_genes]

    groups = make_spatial_groups(xy, config.n_groups, config.seed)
    is_dev, is_test, test_groups = split_dev_test_by_groups(
        groups, config.test_fraction, config.seed
    )
    unique_dev_groups = np.unique(groups[is_dev])
    if len(unique_dev_groups) < 2 or is_test.sum() == 0:
        return pd.DataFrame(), {}, {"receiver_type": receiver_type, "skipped": True}

    x_dev = x_block.to_numpy(float)[is_dev]
    x_test = x_block.to_numpy(float)[is_test]
    y_dev = y_block.to_numpy(float)[is_dev]
    y_test = y_block.to_numpy(float)[is_test]
    groups_dev = groups[is_dev]
    w_dev = weights[is_dev] if weights is not None else None
    w_test = weights[is_test] if weights is not None else None

    n_splits = min(config.n_splits, len(unique_dev_groups))
    gkf = GroupKFold(n_splits=n_splits)
    param_grid = xgb_param_grid(config)
    results = []
    models = {}

    for gi, gene in enumerate(tqdm(y_block.columns, desc=f"XGBoost {receiver_type}"), start=0):
        y_dev_gene = y_dev[:, gi]
        y_test_gene = y_test[:, gi]
        best_score = -np.inf
        best_params = None
        best_n_rounds = None

        for max_depth, eta, subs, colsub, reg_l2, reg_l1 in param_grid:
            params = {
                "max_depth": max_depth,
                "learning_rate": eta,
                "subsample": subs,
                "colsample_bytree": colsub,
                "reg_lambda": reg_l2,
                "reg_alpha": reg_l1,
            }
            fold_scores = []
            nrounds = []
            for tr_idx, va_idx in gkf.split(x_dev, groups=groups_dev):
                model = make_xgb(params, config)
                wtr = w_dev[tr_idx] if w_dev is not None else None
                wva = w_dev[va_idx] if w_dev is not None else None
                model.fit(
                    x_dev[tr_idx],
                    y_dev_gene[tr_idx],
                    sample_weight=wtr,
                    eval_set=[(x_dev[va_idx], y_dev_gene[va_idx])],
                    sample_weight_eval_set=[wva] if wva is not None else None,
                    verbose=False,
                )
                yhat = model.predict(x_dev[va_idx])
                fold_scores.append(r2_weighted(y_dev_gene[va_idx], yhat, wva))
                nrounds.append(
                    int(model.best_iteration) if model.best_iteration is not None else config.n_estimators
                )

            mean_score = float(np.nanmean(fold_scores))
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_n_rounds = int(np.median(nrounds))

        n_dev = x_dev.shape[0]
        n_eval = max(1, int(0.1 * n_dev))
        tr_mask = np.ones(n_dev, dtype=bool)
        tr_mask[-n_eval:] = False
        va_mask = ~tr_mask

        final = make_xgb(best_params, config, n_estimators=max(best_n_rounds or 50, 50))
        final.set_params(
            early_stopping_rounds=min(
                config.early_stop, max(1, (best_n_rounds or config.early_stop) // 3)
            )
        )
        wtr = w_dev[tr_mask] if w_dev is not None else None
        wva = w_dev[va_mask] if w_dev is not None else None
        final.fit(
            x_dev[tr_mask],
            y_dev_gene[tr_mask],
            sample_weight=wtr,
            eval_set=[(x_dev[va_mask], y_dev_gene[va_mask])],
            sample_weight_eval_set=[wva] if wva is not None else None,
            verbose=False,
        )
        yhat_test = final.predict(x_test)
        results.append(
            {
                "receiver_type": receiver_type,
                "gene": gene,
                "cv_mean_r2": float(best_score),
                "test_r2": r2_weighted(y_test_gene, yhat_test, w_test),
                "best_n_estimators": int(
                    final.best_iteration if final.best_iteration is not None else final.n_estimators
                ),
                **best_params,
            }
        )
        models[gene] = final

    summary = pd.DataFrame(results).sort_values(
        ["test_r2", "cv_mean_r2"], ascending=[False, False]
    )
    split_info = {
        "receiver_type": receiver_type,
        "test_groups": test_groups.tolist(),
        "n_cells_model": int(len(x_block)),
        "n_dev": int(is_dev.sum()),
        "n_test": int(is_test.sum()),
        "feature_names": x_block.columns.tolist(),
        "target_genes": y_block.columns.tolist(),
    }
    return summary.reset_index(drop=True), models, split_info


def train_all_receivers(
    df: pd.DataFrame,
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    aux: pd.DataFrame,
    config: PipelineConfig,
):
    all_summaries = []
    all_models = {}
    split_info = {}
    receiver_types = sorted(df[config.class_col].astype(str).unique())
    for receiver_type in receiver_types:
        log(f"Training receiver type: {receiver_type}")
        summary, models, info = train_xgb_for_receiver(
            receiver_type, df, x_receptors, x_exposure, x_cov, y_targets, aux, config
        )
        split_info[receiver_type] = info
        if summary.empty:
            log(f"Skipped {receiver_type}: not enough usable cells/groups/targets.")
            continue
        all_summaries.append(summary)
        all_models[receiver_type] = models
    if not all_summaries:
        raise ValueError("No receiver type produced trained models.")
    return pd.concat(all_summaries, ignore_index=True), all_models, split_info


def permutation_pvalue(observed: float, null_samples: np.ndarray) -> float:
    null = np.asarray(null_samples, float)
    if null.size == 0 or np.isnan(observed):
        return np.nan
    return float((1.0 + np.sum(null >= observed)) / (1.0 + null.size))


def quick_null_for_receiver(
    receiver_type: str,
    genes: list[str],
    df: pd.DataFrame,
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    aux: pd.DataFrame,
    config: PipelineConfig,
    iteration: int,
) -> pd.DataFrame:
    idx_all = df.index[df[config.class_col].astype(str) == receiver_type]
    x_block = pd.concat(
        [x_receptors.loc[idx_all], x_exposure.loc[idx_all], x_cov.loc[idx_all]], axis=1
    )
    y_block = y_targets.loc[idx_all, genes]
    xy = df.loc[idx_all, [config.x_col, config.y_col]].to_numpy(float)
    weights = (
        aux.loc[idx_all, "support_windows"].clip(lower=1).to_numpy()
        if config.use_sample_weights
        else None
    )

    if config.ignore_zero_cov:
        keep = (x_cov.loc[idx_all].sum(axis=1) > 0).to_numpy()
        x_block = x_block.loc[keep]
        y_block = y_block.loc[keep]
        xy = xy[keep]
        if weights is not None:
            weights = weights[keep]
    if len(x_block) < 5:
        return pd.DataFrame()

    groups = make_spatial_groups(xy, config.n_groups, config.seed)
    is_dev, is_test, _ = split_dev_test_by_groups(groups, config.test_fraction, config.seed)
    x_dev = x_block.to_numpy(float)[is_dev]
    x_test = x_block.to_numpy(float)[is_test]
    y_dev = y_block.to_numpy(float)[is_dev]
    y_test = y_block.to_numpy(float)[is_test]
    w_dev = weights[is_dev] if weights is not None else None
    w_test = weights[is_test] if weights is not None else None

    params = {
        "max_depth": 4,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
    }
    n_dev = x_dev.shape[0]
    n_eval = max(1, int(0.1 * n_dev))
    tr_mask = np.ones(n_dev, dtype=bool)
    tr_mask[-n_eval:] = False
    va_mask = ~tr_mask
    rng = np.random.default_rng(config.seed + iteration)
    rows = []
    for gi, gene in enumerate(genes):
        y_perm = y_dev[:, gi].copy()
        rng.shuffle(y_perm)
        model = make_xgb(params, config, n_estimators=min(config.n_estimators, 500))
        model.fit(
            x_dev[tr_mask],
            y_perm[tr_mask],
            sample_weight=w_dev[tr_mask] if w_dev is not None else None,
            eval_set=[(x_dev[va_mask], y_perm[va_mask])],
            sample_weight_eval_set=[w_dev[va_mask]] if w_dev is not None else None,
            verbose=False,
        )
        yhat = model.predict(x_test)
        rows.append(
            {
                "receiver_type": receiver_type,
                "gene": gene,
                "test_r2": r2_weighted(y_test[:, gi], yhat, w_test),
                "iter": iteration,
            }
        )
    return pd.DataFrame(rows)


def compute_pvalues(
    summary_all: pd.DataFrame,
    df: pd.DataFrame,
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    aux: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    if config.n_permutations <= 0:
        return pd.DataFrame()

    null_runs = []
    for it in range(config.n_permutations):
        log(f"Permutation p-value iteration {it + 1}/{config.n_permutations}")
        for receiver_type, sub in summary_all.groupby("receiver_type"):
            genes = sub["gene"].tolist()
            null_df = quick_null_for_receiver(
                receiver_type,
                genes,
                df,
                x_receptors,
                x_exposure,
                x_cov,
                y_targets,
                aux,
                config,
                it,
            )
            if not null_df.empty:
                null_runs.append(null_df)
    if not null_runs:
        return pd.DataFrame()

    null_all = pd.concat(null_runs, ignore_index=True)
    rows = []
    for _, row in summary_all.iterrows():
        null_vals = null_all.loc[
            (null_all["receiver_type"] == row["receiver_type"])
            & (null_all["gene"] == row["gene"]),
            "test_r2",
        ].to_numpy()
        rows.append(
            {
                "receiver_type": row["receiver_type"],
                "gene": row["gene"],
                "test_r2_obs": row["test_r2"],
                "null_mean_r2": float(np.nanmean(null_vals)) if null_vals.size else np.nan,
                "null_std_r2": float(np.nanstd(null_vals)) if null_vals.size else np.nan,
                "n_iter": int(null_vals.size),
                "p_value": permutation_pvalue(row["test_r2"], null_vals),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["receiver_type", "p_value", "test_r2_obs"], ascending=[True, True, False]
    )


def get_receiver_blocks(
    receiver_type: str,
    df: pd.DataFrame,
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    aux: pd.DataFrame,
    config: PipelineConfig,
):
    idx_all = df.index[df[config.class_col].astype(str) == receiver_type]
    x_block = pd.concat(
        [x_receptors.loc[idx_all], x_exposure.loc[idx_all], x_cov.loc[idx_all]], axis=1
    )
    y_block = y_targets.loc[idx_all]
    xy = df.loc[idx_all, [config.x_col, config.y_col]].to_numpy(float)
    weights = (
        aux.loc[idx_all, "support_windows"].clip(lower=1).to_numpy()
        if config.use_sample_weights
        else None
    )
    if config.ignore_zero_cov:
        keep = (x_cov.loc[idx_all].sum(axis=1) > 0).to_numpy()
        x_block = x_block.loc[keep]
        y_block = y_block.loc[keep]
        xy = xy[keep]
        if weights is not None:
            weights = weights[keep]
    if config.target_limit is not None:
        top_genes = y_block.var(axis=0).sort_values(ascending=False).index[: config.target_limit]
        y_block = y_block.loc[:, top_genes]
    return x_block, y_block, xy, weights


def responsive_gene_tables(
    all_models: dict[str, dict[str, XGBRegressor]],
    df: pd.DataFrame,
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    aux: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[str]]]:
    tables = {}
    genes_by_type = {}
    for receiver_type, models in all_models.items():
        x_block, y_block, xy, weights = get_receiver_blocks(
            receiver_type, df, x_receptors, x_exposure, x_cov, y_targets, aux, config
        )
        groups = make_spatial_groups(xy, config.n_groups, config.seed)
        _, is_test, _ = split_dev_test_by_groups(groups, config.test_fraction, config.seed)
        cov_sum = x_cov.loc[x_block.index, :].sum(axis=1).to_numpy()
        test_covpos = is_test & (cov_sum > 0)
        if test_covpos.sum() == 0:
            tables[receiver_type] = pd.DataFrame()
            genes_by_type[receiver_type] = []
            continue
        x_test = x_block.to_numpy(float)[test_covpos]
        y_test = y_block.to_numpy(float)[test_covpos]
        w_test = weights[test_covpos] if weights is not None else None
        rows = []
        for gene, model in models.items():
            if gene not in y_block.columns:
                continue
            y_true = y_test[:, y_block.columns.get_loc(gene)]
            y_pred = model.predict(x_test)
            rows.append(
                {
                    "receiver_type": receiver_type,
                    "gene": gene,
                    "test_r2_covpos": r2_weighted(y_true, y_pred, w_test),
                }
            )
        table = pd.DataFrame(rows).sort_values("test_r2_covpos", ascending=False)
        tables[receiver_type] = table.reset_index(drop=True)
        genes_by_type[receiver_type] = table.loc[table["test_r2_covpos"] > 0, "gene"].tolist()
    return tables, genes_by_type


def run_shap(
    all_models: dict[str, dict[str, XGBRegressor]],
    responsive_tables: dict[str, pd.DataFrame],
    df: pd.DataFrame,
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    aux: pd.DataFrame,
    config: PipelineConfig,
) -> dict[tuple[str, str], pd.DataFrame]:
    if config.skip_shap:
        return {}

    results = {}
    for receiver_type, table in responsive_tables.items():
        if table.empty or receiver_type not in all_models:
            continue
        x_block, _, xy, _ = get_receiver_blocks(
            receiver_type, df, x_receptors, x_exposure, x_cov, y_targets, aux, config
        )
        groups = make_spatial_groups(xy, config.n_groups, config.seed)
        _, is_test, _ = split_dev_test_by_groups(groups, config.test_fraction, config.seed)
        cov_sum = x_cov.loc[x_block.index, :].sum(axis=1).to_numpy()
        test_covpos = is_test & (cov_sum > 0)
        x_test = x_block.to_numpy(float)[test_covpos]
        if x_test.shape[0] == 0:
            continue
        if x_test.shape[0] > config.shap_sample:
            rng = np.random.default_rng(config.seed)
            take = np.sort(rng.choice(x_test.shape[0], size=config.shap_sample, replace=False))
            x_shap = x_test[take]
        else:
            x_shap = x_test

        genes = (
            table.loc[table["test_r2_covpos"] > 0, "gene"]
            .head(config.top_genes_downstream)
            .tolist()
        )
        blocks = (
            ["receptor"] * x_receptors.shape[1]
            + ["ligand_exposure"] * x_exposure.shape[1]
            + ["coverage"] * x_cov.shape[1]
        )
        for gene in tqdm(genes, desc=f"SHAP {receiver_type}"):
            model = all_models[receiver_type].get(gene)
            if model is None:
                continue
            explainer = shap.Explainer(model)
            values = explainer(x_shap)
            mean_abs = np.mean(np.abs(values.values), axis=0)
            results[(receiver_type, gene)] = pd.DataFrame(
                {
                    "feature": x_block.columns,
                    "mean_abs_shap": mean_abs,
                    "block": blocks,
                }
            ).sort_values("mean_abs_shap", ascending=False)
    return results


def safe_weighted_mean(x: np.ndarray, weights: np.ndarray | None) -> float:
    if x.size == 0:
        return np.nan
    if weights is None:
        return float(np.mean(x))
    mask = np.isfinite(x) & np.isfinite(weights)
    x = x[mask]
    weights = weights[mask]
    if x.size == 0 or weights.sum() <= 0:
        return float(np.mean(x)) if x.size else np.nan
    return float(np.sum(weights * x) / weights.sum())


def perturb_signature(
    receiver_type: str,
    genes: list[str],
    mode: str,
    all_models: dict[str, dict[str, XGBRegressor]],
    df: pd.DataFrame,
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    aux: pd.DataFrame,
    config: PipelineConfig,
) -> pd.DataFrame:
    x_block, _, xy, weights = get_receiver_blocks(
        receiver_type, df, x_receptors, x_exposure, x_cov, y_targets, aux, config
    )
    groups = make_spatial_groups(xy, config.n_groups, config.seed)
    _, is_test, _ = split_dev_test_by_groups(groups, config.test_fraction, config.seed)
    cov_sum = x_cov.loc[x_block.index, :].sum(axis=1).to_numpy()
    test_covpos = is_test & (cov_sum > 0)
    x_test = x_block.to_numpy(float)[test_covpos]
    w_test = weights[test_covpos] if weights is not None else None
    if x_test.shape[0] == 0:
        return pd.DataFrame()

    rec_cols = x_receptors.columns.tolist()
    lig_cols = x_exposure.columns.tolist()
    if mode == "ligand_block":
        perturbed = lig_cols
        start = len(rec_cols)
    elif mode == "receptor_KO":
        perturbed = rec_cols
        start = 0
    else:
        raise ValueError("mode must be ligand_block or receptor_KO")

    rows = []
    for gene in genes:
        model = all_models.get(receiver_type, {}).get(gene)
        if model is None:
            continue
        baseline = model.predict(x_test)
        for j, feature in enumerate(perturbed):
            xp = x_test.copy()
            xp[:, start + j] = 0.0
            delta = model.predict(xp) - baseline
            rows.append(
                {
                    "receiver_type": receiver_type,
                    "target_gene": gene,
                    "perturbed_feature": feature,
                    "mode": mode,
                    "delta_mean": float(np.mean(delta)) if delta.size else np.nan,
                    "delta_weighted_mean": safe_weighted_mean(delta, w_test),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("delta_weighted_mean", ascending=False)


def run_perturbations(
    responsive_tables: dict[str, pd.DataFrame],
    all_models: dict[str, dict[str, XGBRegressor]],
    df: pd.DataFrame,
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    aux: pd.DataFrame,
    config: PipelineConfig,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    ligand_signatures = {}
    receptor_signatures = {}
    for receiver_type, table in responsive_tables.items():
        genes = (
            table.loc[table["test_r2_covpos"] > 0, "gene"]
            .head(config.top_genes_downstream)
            .tolist()
        )
        if not genes:
            continue
        ligand_signatures[receiver_type] = perturb_signature(
            receiver_type,
            genes,
            "ligand_block",
            all_models,
            df,
            x_receptors,
            x_exposure,
            x_cov,
            y_targets,
            aux,
            config,
        )
        receptor_signatures[receiver_type] = perturb_signature(
            receiver_type,
            genes,
            "receptor_KO",
            all_models,
            df,
            x_receptors,
            x_exposure,
            x_cov,
            y_targets,
            aux,
            config,
        )
    return ligand_signatures, receptor_signatures


def save_ecdf_plot(summary_all: pd.DataFrame, out_path: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    for receiver_type, sub in summary_all.groupby("receiver_type"):
        values = np.sort(sub["test_r2"].dropna().clip(-1, 1).to_numpy())
        if values.size == 0:
            continue
        ecdf = np.arange(1, values.size + 1) / values.size
        plt.step(values, ecdf, where="post", label=str(receiver_type), lw=1.5)
    plt.xlabel(r"$R^2$")
    plt.ylabel("Fraction of genes (ECDF)")
    plt.title("Per-gene $R^2$ ECDF by receiver cell type")
    plt.xlim(-0.2, 1.0)
    plt.ylim(0, 1.0)
    plt.grid(alpha=0.35)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_outputs(
    config: PipelineConfig,
    results_df: pd.DataFrame,
    island_index: pd.DataFrame,
    binary_mat: pd.DataFrame,
    aux: pd.DataFrame,
    lr_pairs_kept: pd.DataFrame,
    lr_report: dict[str, Any],
    x_receptors: pd.DataFrame,
    x_exposure: pd.DataFrame,
    x_cov: pd.DataFrame,
    y_targets: pd.DataFrame,
    summary_all: pd.DataFrame,
    all_models: dict[str, dict[str, XGBRegressor]],
    split_info: dict[str, Any],
    pvalues: pd.DataFrame,
    responsive_tables: dict[str, pd.DataFrame],
    shap_results: dict[tuple[str, str], pd.DataFrame],
    ligand_signatures: dict[str, pd.DataFrame],
    receptor_signatures: dict[str, pd.DataFrame],
) -> None:
    results_dir = ensure_dir(config.results_dir)
    ensure_dir(results_dir / "features")
    ensure_dir(results_dir / "downstream")

    results_df.to_csv(results_dir / "pairwise_weighted_pcc_map.csv", index=False)
    island_index.to_csv(results_dir / "island_index.csv", index=False)
    binary_mat.to_csv(results_dir / "binary_colocalization.csv")
    aux.to_csv(results_dir / "cell_coverage.csv")
    lr_pairs_kept.to_csv(results_dir / "lr_pairs_kept.csv", index=False)
    (results_dir / "lr_feature_report.json").write_text(json.dumps(lr_report, indent=2))

    x_receptors.to_csv(results_dir / "features" / "x_receptors.csv")
    x_exposure.to_csv(results_dir / "features" / "x_ligand_exposure.csv")
    x_cov.to_csv(results_dir / "features" / "x_coverage.csv")
    pd.Series(y_targets.columns, name="target_gene").to_csv(
        results_dir / "features" / "target_genes.csv", index=False
    )

    summary_all.to_csv(results_dir / "r2_summary.csv", index=False)
    save_ecdf_plot(summary_all, results_dir / "downstream" / "r2_ecdf.png")
    (results_dir / "split_info.json").write_text(json.dumps(split_info, indent=2))

    if not pvalues.empty:
        pvalues.to_csv(results_dir / "downstream" / "p_values.csv", index=False)

    responsive_dir = ensure_dir(results_dir / "downstream" / "responsive_genes")
    for receiver_type, table in responsive_tables.items():
        if not table.empty:
            table.to_csv(responsive_dir / f"{safe_name(receiver_type)}.csv", index=False)

    shap_dir = ensure_dir(results_dir / "downstream" / "shap")
    for (receiver_type, gene), table in shap_results.items():
        table.to_csv(shap_dir / f"{safe_name(receiver_type)}__{safe_name(gene)}.csv", index=False)

    ligand_dir = ensure_dir(results_dir / "downstream" / "perturbations" / "ligands")
    receptor_dir = ensure_dir(results_dir / "downstream" / "perturbations" / "receptors")
    for receiver_type, table in ligand_signatures.items():
        if not table.empty:
            table.to_csv(ligand_dir / f"{safe_name(receiver_type)}.csv", index=False)
    for receiver_type, table in receptor_signatures.items():
        if not table.empty:
            table.to_csv(receptor_dir / f"{safe_name(receiver_type)}.csv", index=False)

    if config.save_models:
        model_dir = ensure_dir(results_dir / "models")
        for receiver_type, models in all_models.items():
            receiver_dir = ensure_dir(model_dir / safe_name(receiver_type))
            for gene, model in models.items():
                model.save_model(str(receiver_dir / f"{safe_name(gene)}.json"))
        joblib.dump(all_models, results_dir / "models" / "all_models.joblib")

    config_dict = asdict(config)
    config_dict["csv_path"] = str(config.csv_path)
    config_dict["lr_pairs_path"] = str(config.lr_pairs_path)
    config_dict["results_dir"] = str(config.results_dir)
    (results_dir / "config.json").write_text(json.dumps(config_dict, indent=2))


def run_pipeline(config: PipelineConfig) -> None:
    ensure_dir(config.results_dir)
    log(f"Reading expression CSV: {config.csv_path}")
    df = load_expression_table(config)
    log(f"Loaded {df.shape[0]} cells and {df.shape[1]} columns.")

    log("Computing weighted PCC map.")
    results_df = compute_pairwise_weighted_pcc(df, config)
    log(f"Weighted PCC rows: {len(results_df)}")

    log("Finding colocalized islands.")
    islands, island_index = find_islands_for_all_pairs(
        results_df,
        r_threshold=config.island_r_threshold,
        min_windows=config.island_min_windows,
    )
    log(f"Detected {len(islands)} islands.")

    log("Encoding per-cell island coverage.")
    binary_mat, aux = encode_cell_colocalization(df, results_df, islands, config)

    log(f"Reading ligand-receptor pairs: {config.lr_pairs_path}")
    lr_pairs = pd.read_csv(config.lr_pairs_path)
    x_receptors, x_ligands, lr_pairs_kept, lr_report = prepare_lr_features(
        df, lr_pairs, config
    )
    log(
        "Matched "
        f"{lr_report['n_pairs_kept']} LR pairs, "
        f"{lr_report['n_unique_receptors_kept']} receptors, "
        f"{lr_report['n_unique_ligands_kept']} ligands."
    )

    log(f"Computing ligand exposure with mode={config.exposure_mode}.")
    x_exposure = compute_ligand_exposure(
        df, x_ligands, results_df, islands, aux, config
    )
    x_cov = make_coverage_features(aux)

    log("Filtering target genes.")
    y_targets = make_targets(df, lr_pairs_kept, config)
    log(f"Targets kept: {y_targets.shape[1]}")

    log("Training XGBoost models.")
    summary_all, all_models, split_info = train_all_receivers(
        df, x_receptors, x_exposure, x_cov, y_targets, aux, config
    )

    log("Computing permutation p-values.")
    pvalues = compute_pvalues(
        summary_all, df, x_receptors, x_exposure, x_cov, y_targets, aux, config
    )

    log("Finding LR-responsive genes.")
    responsive_tables, _ = responsive_gene_tables(
        all_models, df, x_receptors, x_exposure, x_cov, y_targets, aux, config
    )

    log("Running SHAP analysis.")
    shap_results = run_shap(
        all_models,
        responsive_tables,
        df,
        x_receptors,
        x_exposure,
        x_cov,
        y_targets,
        aux,
        config,
    )

    log("Running in-silico perturbations.")
    ligand_signatures, receptor_signatures = run_perturbations(
        responsive_tables,
        all_models,
        df,
        x_receptors,
        x_exposure,
        x_cov,
        y_targets,
        aux,
        config,
    )

    log("Saving results.")
    save_outputs(
        config,
        results_df,
        island_index,
        binary_mat,
        aux,
        lr_pairs_kept,
        lr_report,
        x_receptors,
        x_exposure,
        x_cov,
        y_targets,
        summary_all,
        all_models,
        split_info,
        pvalues,
        responsive_tables,
        shap_results,
        ligand_signatures,
        receptor_signatures,
    )
    log(f"Done. Results written to {config.results_dir}")


def default_lr_pairs_path() -> Path:
    return Path(__file__).resolve().parents[1] / "data" / "mouse_850_lr_pairs_cpdb_interactions.csv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the ColocEM weighted-PCC + XGBoost pipeline."
    )
    parser.add_argument("csv_path", type=Path, help="Path to the cell-by-gene expression CSV.")
    parser.add_argument(
        "--lr-pairs",
        type=Path,
        default=default_lr_pairs_path(),
        help="Ligand-receptor pairs CSV.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory where pipeline outputs will be written.",
    )
    parser.add_argument("--x-col", default="x")
    parser.add_argument("--y-col", default="y")
    parser.add_argument("--class-col", default="class")
    parser.add_argument("--cell-label-col", default="cell_label")
    parser.add_argument("--ligand-col", default="ligand_genesymbol")
    parser.add_argument("--receptor-col", default="target_genesymbol")
    parser.add_argument(
        "--meta-cols",
        default="x,y,class,cell_label",
        help="Comma-separated non-gene columns to exclude from gene matrices.",
    )
    parser.add_argument("--gene-start", type=int, default=None)
    parser.add_argument("--gene-end", type=int, default=None)

    parser.add_argument("--win-size", type=float, default=2.0)
    parser.add_argument("--grid-n", type=int, default=25)
    parser.add_argument("--kde-bw", default="0.2")
    parser.add_argument("--min-kde-points", type=int, default=5)
    parser.add_argument("--weight-mode", choices=["sum", "prod"], default="sum")
    parser.add_argument("--island-r-threshold", type=float, default=0.7)
    parser.add_argument("--island-min-windows", type=int, default=4)
    parser.add_argument("--coverage-theta", type=float, default=0.5)
    parser.add_argument("--coverage-min-support", type=int, default=3)
    parser.add_argument("--exposure-mode", choices=["mean", "kde"], default="kde")
    parser.add_argument("--exposure-sigma", type=float, default=None)

    parser.add_argument("--min-detect-frac", type=float, default=0.01)
    parser.add_argument("--min-var-quantile", type=float, default=0.05)
    parser.add_argument("--keep-ligands", action="store_true")
    parser.add_argument("--keep-technicals", action="store_true")

    parser.add_argument("--n-groups", type=int, default=8)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--target-limit", type=int, default=300)
    parser.add_argument("--all-targets", action="store_true")
    parser.add_argument("--keep-zero-cov", action="store_true")
    parser.add_argument("--no-sample-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--early-stop", type=int, default=100)
    parser.add_argument("--tree-method", default="hist")
    parser.add_argument("--n-jobs", type=int, default=0)
    parser.add_argument("--max-depths", default="4,6")
    parser.add_argument("--learning-rates", default="0.03,0.1")
    parser.add_argument("--subsample", default="0.8,1.0")
    parser.add_argument("--colsample", default="0.8,1.0")
    parser.add_argument("--reg-l2", default="1,5,10")
    parser.add_argument("--reg-l1", default="0,1")

    parser.add_argument("--n-permutations", type=int, default=5)
    parser.add_argument("--top-genes-downstream", type=int, default=50)
    parser.add_argument("--shap-sample", type=int, default=5000)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--no-save-models", action="store_true")
    return parser


def parse_kde_bw(value: str) -> float | str:
    if value in {"scott", "silverman"}:
        return value
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--kde-bw must be a float, 'scott', or 'silverman'."
        ) from exc


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    target_limit = None if args.all_targets else args.target_limit
    return PipelineConfig(
        csv_path=args.csv_path,
        lr_pairs_path=args.lr_pairs,
        results_dir=args.results_dir,
        x_col=args.x_col,
        y_col=args.y_col,
        class_col=args.class_col,
        cell_label_col=args.cell_label_col,
        ligand_col=args.ligand_col,
        receptor_col=args.receptor_col,
        meta_cols=tuple(c.strip() for c in args.meta_cols.split(",") if c.strip()),
        gene_start=args.gene_start,
        gene_end=args.gene_end,
        win_size=args.win_size,
        grid_n=args.grid_n,
        kde_bw=parse_kde_bw(args.kde_bw),
        min_kde_points=args.min_kde_points,
        weight_mode=args.weight_mode,
        island_r_threshold=args.island_r_threshold,
        island_min_windows=args.island_min_windows,
        coverage_theta=args.coverage_theta,
        coverage_min_support=args.coverage_min_support,
        exposure_mode=args.exposure_mode,
        exposure_sigma=args.exposure_sigma,
        min_detect_frac=args.min_detect_frac,
        min_var_quantile=args.min_var_quantile,
        drop_ligands=not args.keep_ligands,
        drop_technicals=not args.keep_technicals,
        n_groups=args.n_groups,
        test_fraction=args.test_fraction,
        n_splits=args.n_splits,
        target_limit=target_limit,
        ignore_zero_cov=not args.keep_zero_cov,
        use_sample_weights=not args.no_sample_weights,
        seed=args.seed,
        n_estimators=args.n_estimators,
        early_stop=args.early_stop,
        tree_method=args.tree_method,
        n_jobs=args.n_jobs,
        max_depths=parse_number_list(args.max_depths, int),
        learning_rates=parse_number_list(args.learning_rates, float),
        subsample=parse_number_list(args.subsample, float),
        colsample=parse_number_list(args.colsample, float),
        reg_l2=parse_number_list(args.reg_l2, float),
        reg_l1=parse_number_list(args.reg_l1, float),
        n_permutations=args.n_permutations,
        top_genes_downstream=args.top_genes_downstream,
        shap_sample=args.shap_sample,
        skip_shap=args.skip_shap,
        save_models=not args.no_save_models,
    )


def main() -> None:
    args = build_parser().parse_args()
    config = config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
