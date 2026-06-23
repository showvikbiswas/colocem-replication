#!/usr/bin/env python3
"""Compare Squidpy neighborhood enrichment with ColocEM pair scores."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path


LOGGER = logging.getLogger("squidpy_colocem_compare")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge Squidpy z-scores with ColocEM pair scores and compute comparison metrics."
    )
    parser.add_argument(
        "--squidpy-long",
        type=Path,
        required=True,
        help="Squidpy long-format CSV with cell_type_1, cell_type_2, squidpy_zscore.",
    )
    parser.add_argument(
        "--colocem-islands",
        type=Path,
        required=True,
        help="ColocEM island_index.csv with class_A, class_B, and score columns.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/squidpy_colocem_comparison"),
        help="Directory where merged tables, metrics, and plots will be written.",
    )
    parser.add_argument(
        "--top-k",
        default="10,20,50",
        help="Comma-separated top-k values for overlap/Jaccard metrics.",
    )
    parser.add_argument(
        "--include-self-pairs",
        action="store_true",
        help="Include self-pairs such as A-A. By default they are removed.",
    )
    parser.add_argument(
        "--squidpy-aggregate",
        choices=("mean", "max", "min"),
        default="mean",
        help="How to aggregate reciprocal Squidpy directed pair scores.",
    )
    parser.add_argument(
        "--colocem-score-column",
        default="cluster_mass_z",
        help="ColocEM island column to aggregate into the primary colocem_score.",
    )
    parser.add_argument(
        "--colocem-aggregate",
        choices=("sum", "mean", "max", "min"),
        default="sum",
        help="How to aggregate multiple ColocEM islands per pair.",
    )
    return parser


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def parse_top_k(value: str) -> list[int]:
    try:
        top_k = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise ValueError("--top-k must be a comma-separated list of positive integers.") from exc
    if not top_k or any(k < 1 for k in top_k):
        raise ValueError("--top-k must contain at least one positive integer.")
    return sorted(set(top_k))


def require_columns(df: pd.DataFrame, required: list[str], source: Path) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required column(s): {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def add_pair_keys(
    df: pd.DataFrame,
    left_col: str,
    right_col: str,
    include_self_pairs: bool,
) -> pd.DataFrame:
    import numpy as np

    out = df.copy()
    left = out[left_col].astype(str)
    right = out[right_col].astype(str)
    out["pair_1"] = np.minimum(left, right)
    out["pair_2"] = np.maximum(left, right)
    if not include_self_pairs:
        before = len(out)
        out = out.loc[out["pair_1"] != out["pair_2"]].copy()
        LOGGER.info("Dropped %d self-pair row(s).", before - len(out))
    return out


def aggregate_series(grouped, column: str, method: str):
    if method == "sum":
        return grouped[column].sum()
    if method == "mean":
        return grouped[column].mean()
    if method == "max":
        return grouped[column].max()
    if method == "min":
        return grouped[column].min()
    raise ValueError(f"Unsupported aggregate method: {method}")


def load_squidpy_pairs(
    path: Path,
    aggregate: str,
    include_self_pairs: bool,
) -> pd.DataFrame:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Squidpy long-format CSV does not exist: {path}")
    df = pd.read_csv(path)
    require_columns(df, ["cell_type_1", "cell_type_2", "squidpy_zscore"], path)
    df["squidpy_zscore"] = pd.to_numeric(df["squidpy_zscore"], errors="coerce")
    df = add_pair_keys(df, "cell_type_1", "cell_type_2", include_self_pairs)
    grouped = df.groupby(["pair_1", "pair_2"])
    agg = aggregate_series(grouped, "squidpy_zscore", aggregate).rename("squidpy_zscore")
    agg = agg.reset_index()
    sizes = grouped.size().rename("squidpy_n_entries").reset_index()
    agg = agg.merge(sizes, on=["pair_1", "pair_2"], how="left")
    LOGGER.info("Loaded %d unordered Squidpy pair score(s).", len(agg))
    return agg


def load_colocem_pairs(
    path: Path,
    score_column: str,
    aggregate: str,
    include_self_pairs: bool,
) -> pd.DataFrame:
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"ColocEM island index CSV does not exist: {path}")
    df = pd.read_csv(path)
    require_columns(df, ["class_A", "class_B", score_column], path)
    df[score_column] = pd.to_numeric(df[score_column], errors="coerce")
    df = add_pair_keys(df, "class_A", "class_B", include_self_pairs)
    grouped = df.groupby(["pair_1", "pair_2"])

    score = aggregate_series(grouped, score_column, aggregate).rename("colocem_score")
    score = score.reset_index()
    score["colocem_score_column"] = score_column
    score["colocem_aggregate"] = aggregate
    sizes = grouped.size().rename("colocem_n_islands").reset_index()
    score = score.merge(sizes, on=["pair_1", "pair_2"], how="left")

    if "median_r" in df.columns:
        df["median_r"] = pd.to_numeric(df["median_r"], errors="coerce")
        med_grouped = df.groupby(["pair_1", "pair_2"])
        secondary = med_grouped["median_r"].agg(
            colocem_max_median_r="max",
            colocem_mean_median_r="mean",
        )
        score = score.merge(secondary.reset_index(), on=["pair_1", "pair_2"], how="left")
    if "n_windows" in df.columns:
        df["n_windows"] = pd.to_numeric(df["n_windows"], errors="coerce")
        windows = (
            df.groupby(["pair_1", "pair_2"])["n_windows"]
            .sum()
            .rename("colocem_total_windows")
            .reset_index()
        )
        score = score.merge(windows, on=["pair_1", "pair_2"], how="left")

    LOGGER.info("Loaded %d unordered ColocEM pair score(s).", len(score))
    return score


def compute_summary_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    import pandas as pd
    from scipy.stats import pearsonr, spearmanr

    valid = merged[["squidpy_zscore", "colocem_score"]].replace([np.inf, -np.inf], np.nan).dropna()
    n_pairs = len(valid)
    if n_pairs < 2:
        spearman_r = spearman_p = pearson_r = pearson_p = np.nan
    else:
        spearman_r, spearman_p = spearmanr(valid["squidpy_zscore"], valid["colocem_score"])
        pearson_r, pearson_p = pearsonr(valid["squidpy_zscore"], valid["colocem_score"])

    return pd.DataFrame(
        [
            {
                "n_merged_pairs": len(merged),
                "n_finite_pairs": n_pairs,
                "spearman_r": spearman_r,
                "spearman_p_value": spearman_p,
                "pearson_r": pearson_r,
                "pearson_p_value": pearson_p,
            }
        ]
    )


def pair_set(df: pd.DataFrame, k: int, score_col: str) -> set[tuple[str, str]]:
    import numpy as np

    ranked = (
        df[["pair_1", "pair_2", score_col]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[score_col])
        .sort_values(score_col, ascending=False)
        .head(k)
    )
    return set(zip(ranked["pair_1"], ranked["pair_2"]))


def compute_topk_metrics(merged: pd.DataFrame, top_k: list[int]) -> pd.DataFrame:
    import numpy as np
    import pandas as pd

    rows = []
    valid = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["squidpy_zscore", "colocem_score"]
    )
    for k in top_k:
        squidpy_top = pair_set(valid, k, "squidpy_zscore")
        colocem_top = pair_set(valid, k, "colocem_score")
        intersection = squidpy_top & colocem_top
        union = squidpy_top | colocem_top
        rows.append(
            {
                "k": k,
                "n_squidpy_top": len(squidpy_top),
                "n_colocem_top": len(colocem_top),
                "overlap_count": len(intersection),
                "overlap_fraction_of_k": len(intersection) / k,
                "jaccard_similarity": len(intersection) / len(union) if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def save_scatter(merged: pd.DataFrame, summary: pd.DataFrame, output_path: Path) -> None:
    import numpy as np

    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = merged.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["squidpy_zscore", "colocem_score"]
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(valid["squidpy_zscore"], valid["colocem_score"], s=28, alpha=0.75)
    ax.set_xlabel("Squidpy neighborhood-enrichment z-score")
    ax.set_ylabel("ColocEM aggregate island score")
    if not summary.empty:
        row = summary.iloc[0]
        title = (
            f"Spearman ρ={row['spearman_r']:.3g}, "
            f"n={int(row['n_finite_pairs'])}"
        )
    else:
        title = "Squidpy vs ColocEM pair scores"
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    top_k = parse_top_k(args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    squidpy = load_squidpy_pairs(
        args.squidpy_long,
        aggregate=args.squidpy_aggregate,
        include_self_pairs=args.include_self_pairs,
    )
    colocem = load_colocem_pairs(
        args.colocem_islands,
        score_column=args.colocem_score_column,
        aggregate=args.colocem_aggregate,
        include_self_pairs=args.include_self_pairs,
    )
    merged = squidpy.merge(colocem, on=["pair_1", "pair_2"], how="inner")
    if merged.empty:
        raise ValueError("No overlapping cell-type pairs were found between Squidpy and ColocEM.")

    summary = compute_summary_metrics(merged)
    topk = compute_topk_metrics(merged, top_k)

    merged_path = args.output_dir / "merged_squidpy_colocem_pairs.csv"
    summary_path = args.output_dir / "summary_metrics.csv"
    topk_path = args.output_dir / "topk_overlap.csv"
    plot_path = args.output_dir / "squidpy_vs_colocem_scatter.pdf"

    merged.to_csv(merged_path, index=False)
    summary.to_csv(summary_path, index=False)
    topk.to_csv(topk_path, index=False)
    save_scatter(merged, summary, plot_path)

    LOGGER.info("Saved merged pair table: %s", merged_path)
    LOGGER.info("Saved summary metrics: %s", summary_path)
    LOGGER.info("Saved top-k metrics: %s", topk_path)
    LOGGER.info("Saved scatter plot: %s", plot_path)


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        LOGGER.exception("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
