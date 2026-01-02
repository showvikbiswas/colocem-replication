# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.metrics import r2_score
from itertools import product
import re


# %%
allexp = pd.read_csv('data/allexp.csv')

# %%
allexp['class'].unique()

# %%
# plot all clasases by x and y coordinates using pyplot
plt.figure(figsize=(10, 8))
classes = allexp['class'].unique()
colors = plt.cm.get_cmap('tab10', len(classes))
for i, class_name in enumerate(classes):
    if class_name != '04 DG-IMN Glut':
        continue
    class_data = allexp[allexp['class'] == class_name]
    plt.scatter(class_data['x'], class_data['y'], color=colors(i), label=class_name, s=1)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Scatter Plot of Classes by X and Y Coordinates')
plt.legend()
plt.show()

# %%
cell_types = allexp['class'].unique()

for ctype in cell_types:
    sub = allexp[allexp['class'] == ctype]
    x, y = sub['x'].values, sub['y'].values
    
    # KDE
    values = np.vstack([x, y])
    kde = gaussian_kde(values, bw_method=0.2)  # tweak bw_method
    
    # Evaluate on a grid
    X, Y = np.mgrid[allexp['x'].min():allexp['x'].max():100j,
                    allexp['y'].min():allexp['y'].max():100j]
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    # Plot
    plt.figure()
    plt.contourf(X, Y, Z, cmap="viridis")
    plt.title(f"KDE for {ctype}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="Density")
    plt.show()

# %% [markdown]
# # Unweighted PCC

# %% [markdown]
# ## Helper Functions

# %%
win_size = 2.0        # sliding window side length (in your coordinate units)
grid_n   = 25           # grid resolution inside each window (grid_n x grid_n)
bw       = 0.2          # KDE bandwidth for scipy gaussian_kde (float or 'scott'/'silverman')
min_pts  = 5            # minimum points to fit a KDE for a class
weight_mode = 'sum'  # 'sum' -> w_i = a_i + b_i ; 'prod' -> w_i = a_i * b_i
eps        = 1e-12   # numerical floor to avoid 0-division

# %%
from itertools import combinations
from scipy.stats import gaussian_kde, pearsonr

def build_global_kdes(df, bw_method=bw, min_points=min_pts):
    """
    Build a gaussian_kde per class using ALL points (global KDEs).
    Returns: dict[class] -> fitted KDE
    """
    kdes = {}
    for ctype, sub in df.groupby('class'):
        pts = sub[['x','y']].to_numpy()
        if pts.shape[0] < min_points:
            # Not enough points to build a stable KDE; skip this class
            continue
        kdes[ctype] = gaussian_kde(pts.T, bw_method=bw_method)
    return kdes

def sliding_windows(xmin, xmax, ymin, ymax, size):
    """
    Yield windows (x0, x1, y0, y1) with stride size/2, covering [xmin,xmax]x[ymin,ymax].
    The last window is clipped to remain within bounds.
    """
    step = size / 2.0

    def edges(a_min, a_max):
        starts = []
        cur = a_min
        while cur + size <= a_max + 1e-9:
            starts.append(cur)
            cur += step
        # ensure we end exactly at the boundary
        if len(starts) == 0 or starts[-1] + size < a_max:
            starts.append(max(a_min, a_max - size))
        return sorted(set(starts))

    xs = edges(xmin, xmax)
    ys = edges(ymin, ymax)

    for x0 in xs:
        for y0 in ys:
            yield (x0, x0 + size, y0, y0 + size)

def grid_cell_centers(x0, x1, y0, y1, n):
    """
    Return (GX, GY) mesh of grid cell centers (n x n) within the window [x0,x1]x[y0,y1].
    """
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n
    xs = x0 + hx * (np.arange(n) + 0.5)
    ys = y0 + hy * (np.arange(n) + 0.5)
    GX, GY = np.meshgrid(xs, ys, indexing='xy')
    return GX, GY, hx, hy

def windowwise_normalize(Z, hx, hy):
    """
    Normalize a KDE patch so its discrete integral over the window is 1.
    Z is n x n over centers; integral ≈ Z.sum() * hx * hy.
    """
    mass = Z.sum() * hx * hy
    return Z / mass if mass > 0 else Z

def compute_pairwise_pcc_map(df):
    """
    Main driver:
      - builds global KDEs per class
      - iterates sliding windows
      - for each class pair, evaluates KDEs on window grid, normalizes within window, computes PCC
      - returns a results DataFrame
    """
    # Bounds for windows
    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()

    # Global KDEs
    kdes = build_global_kdes(df)
    classes = sorted(kdes.keys())
    pairs = list(combinations(classes, 2))

    records = []

    # Slide windows
    for (x0, x1, y0, y1) in sliding_windows(xmin, xmax, ymin, ymax, win_size):
        # grid of cell centers for this window
        GX, GY, hx, hy = grid_cell_centers(x0, x1, y0, y1, grid_n)
        XY = np.vstack([GX.ravel(), GY.ravel()])

        for A, B in pairs:
            kdeA = kdes.get(A)
            kdeB = kdes.get(B)
            if kdeA is None or kdeB is None:
                # One of the KDEs wasn't available (too few points)
                continue

            # Evaluate KDEs on grid centers inside this window
            ZA = kdeA(XY).reshape(GX.shape)
            ZB = kdeB(XY).reshape(GX.shape)

            # Window-wise normalization (probability over the window)
            ZA = windowwise_normalize(ZA, hx, hy)
            ZB = windowwise_normalize(ZB, hx, hy)

            # Pearson correlation across grid cells (flatten)
            a = ZA.ravel()
            b = ZB.ravel()

            # If either map is constant, pearsonr is undefined; handle gracefully
            if np.allclose(a, a[0]) or np.allclose(b, b[0]):
                r = np.nan
                p = np.nan
            else:
                r, p = pearsonr(a, b)

            records.append({
                'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                'class_A': A, 'class_B': B,
                'pearson_r': r, 'p_value': p,
                'grid_n': grid_n
            })

    return pd.DataFrame.from_records(records)

# %% [markdown]
# ## PCC calcuation

# %%
results_df = compute_pairwise_pcc_map(allexp)
results_df.head()

# %% [markdown]
# # Weighted PCC

# %% [markdown]
# ## Helper Functions

# %%
from itertools import combinations
from scipy.stats import gaussian_kde, pearsonr

def build_global_kdes(df, bw_method=bw, min_points=min_pts):
    """Fit a global gaussian_kde per class using all points (avoids edge bias)."""
    kdes = {}
    for ctype, sub in df.groupby('class'):
        pts = sub[['x','y']].to_numpy()
        if pts.shape[0] >= min_points:
            kdes[ctype] = gaussian_kde(pts.T, bw_method=bw_method)
    return kdes

def sliding_windows(xmin, xmax, ymin, ymax, size):
    """Yield (x0,x1,y0,y1) with stride size/2, clipped to bounds."""
    step = size / 2.0
    def starts(lo, hi):
        s = []
        cur = lo
        while cur + size <= hi + 1e-9:
            s.append(cur); cur += step
        if not s or s[-1] + size < hi:  # ensure coverage to the edge
            s.append(max(lo, hi - size))
        return sorted(set(s))
    for x0 in starts(xmin, xmax):
        for y0 in starts(ymin, ymax):
            yield (x0, x0 + size, y0, y0 + size)

def grid_cell_centers(x0, x1, y0, y1, n):
    """n x n mesh of grid cell centers inside the window + cell sizes (hx, hy)."""
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n
    xs = x0 + hx * (np.arange(n) + 0.5)
    ys = y0 + hy * (np.arange(n) + 0.5)
    GX, GY = np.meshgrid(xs, ys, indexing='xy')
    return GX, GY, hx, hy

def windowwise_normalize(Z, hx, hy):
    """Rescale KDE patch so its discrete integral in the window is 1."""
    mass = Z.sum() * hx * hy
    return Z / mass if mass > 0 else Z

def weighted_pearson(a, b, mode='sum', eps=1e-12):
    """
    Weighted Pearson correlation between arrays a, b (same shape).
    mode='sum' uses w=a+b; mode='prod' uses w=a*b.
    Returns np.nan if variance is zero or total weight ~0.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()

    if mode == 'prod':
        w = a * b
    else:
        w = a + b

    w_sum = w.sum()
    if not np.isfinite(w_sum) or w_sum <= eps:
        return np.nan

    w = w / w_sum  # normalize weights to sum to 1 (probability weights)

    mu_a = (w * a).sum()
    mu_b = (w * b).sum()

    da = a - mu_a
    db = b - mu_b

    var_a = (w * da * da).sum()
    var_b = (w * db * db).sum()
    if var_a <= eps or var_b <= eps:
        return np.nan

    cov_ab = (w * da * db).sum()
    return cov_ab / np.sqrt(var_a * var_b)

def compute_pairwise_weighted_pcc(df):
    """
    - Fit global KDEs per class.
    - Slide window; make a grid of centers; evaluate KDEs.
    - Window-wise normalize each KDE patch.
    - Compute **weighted** Pearson r for each class pair in each window.
    """
    xmin, xmax = df['x'].min(), df['x'].max()
    ymin, ymax = df['y'].min(), df['y'].max()

    kdes = build_global_kdes(df)
    classes = sorted(kdes.keys())
    pairs = list(combinations(classes, 2))

    records = []
    for (x0, x1, y0, y1) in sliding_windows(xmin, xmax, ymin, ymax, win_size):
        GX, GY, hx, hy = grid_cell_centers(x0, x1, y0, y1, grid_n)
        XY = np.vstack([GX.ravel(), GY.ravel()])

        for A, B in pairs:
            kdeA = kdes.get(A); kdeB = kdes.get(B)
            if kdeA is None or kdeB is None:
                continue

            ZA = kdeA(XY).reshape(GX.shape)
            ZB = kdeB(XY).reshape(GX.shape)

            # Window-wise normalization (probability maps over this window)
            ZA = windowwise_normalize(ZA, hx, hy)
            ZB = windowwise_normalize(ZB, hx, hy)

            # Weighted PCC so low-signal cells don't dominate
            r_w = weighted_pearson(ZA, ZB, mode=weight_mode, eps=eps)

            records.append({
                'x0': x0, 'x1': x1, 'y0': y0, 'y1': y1,
                'class_A': A, 'class_B': B,
                'weighted_pearson_r': r_w,
                'grid_n': grid_n, 'win_size': win_size,
                'weight_mode': weight_mode
            })

    return pd.DataFrame.from_records(records)


# %% [markdown]
# ## PCC Calculation

# %%
results_df = compute_pairwise_weighted_pcc(allexp)
results_df.head()

# %% [markdown]
# # Island Formation

# %%
import numpy as np
import pandas as pd
from scipy import ndimage

# -------------------------------
# Helpers
# -------------------------------

def fisher_z(r, eps=1e-12):
    """Fisher z-transform with safe clipping."""
    r = np.asarray(r, dtype=float)
    r = np.where(np.isfinite(r), r, np.nan)
    r = np.clip(r, -1 + eps, 1 - eps)
    # if r is a scalar, np.arctanh returns a scalar
    # make r an array to ensure z is also an array
    z = np.arctanh(r)
    # if z is infinite, set to nan
    if np.isinf(z):
        z = np.nan
    return z

def _build_pair_grids(results_df, A, B, r_col='weighted_pearson_r'):
    """
    From a long-form results_df -> 2D grids aligned on unique (x0,y0).
    Returns dict with r_grid, z_grid, and x0/x1/y0/y1 grids.
    """
    sub = results_df[(results_df['class_A'] == A) & (results_df['class_B'] == B)].copy()
    if sub.empty:
        return None

    # Ensure uniqueness if code was run multiple times
    sub = sub.drop_duplicates(subset=['x0','y0','x1','y1','class_A','class_B'])

    xs = np.array(sorted(sub['x0'].unique()))
    ys = np.array(sorted(sub['y0'].unique()))

    ix = {x0:i for i,x0 in enumerate(xs)}
    iy = {y0:i for i,y0 in enumerate(ys)}

    H, W = len(ys), len(xs)
    r_grid  = np.full((H, W), np.nan, dtype=float)
    z_grid  = np.full((H, W), np.nan, dtype=float)
    x0_grid = np.full((H, W), np.nan, dtype=float)
    x1_grid = np.full((H, W), np.nan, dtype=float)
    y0_grid = np.full((H, W), np.nan, dtype=float)
    y1_grid = np.full((H, W), np.nan, dtype=float)

    for _, row in sub.iterrows():
        i = iy[row['y0']]
        j = ix[row['x0']]
        r = row.get(r_col, np.nan)
        r_grid[i, j] = r
        z_grid[i, j] = fisher_z(r)
        x0_grid[i, j] = row['x0']; x1_grid[i, j] = row['x1']
        y0_grid[i, j] = row['y0']; y1_grid[i, j] = row['y1']

    return {
        'xs': xs, 'ys': ys,
        'r_grid': r_grid, 'z_grid': z_grid,
        'x0_grid': x0_grid, 'x1_grid': x1_grid,
        'y0_grid': y0_grid, 'y1_grid': y1_grid
    }

def _label_8_connected(mask):
    """8-connected component labeling (NaNs already excluded in mask)."""
    structure = np.ones((3,3), dtype=int)   # 8-connectivity
    labels, n = ndimage.label(mask, structure=structure)
    return labels, n

# -------------------------------
# Main: find islands for all pairs
# -------------------------------

def find_islands_for_all_pairs(
    results_df,
    r_threshold=0.5,           # threshold in r; internally converted to z
    r_col='weighted_pearson_r',
    min_windows=1              # drop tiny islands if desired
):
    """
    Build 8-connected 'islands' per (class_A, class_B).
    Returns: list of island dicts + an index table for quick lookup.
    """
    islands = []
    index_rows = []

    z_thr = float(fisher_z(r_threshold))

    pairs = (
        results_df[['class_A','class_B']]
        .drop_duplicates()
        .sort_values(['class_A','class_B'])
        .itertuples(index=False, name=None)
    )

    for (A, B) in pairs:
        grids = _build_pair_grids(results_df, A, B, r_col=r_col)
        if grids is None:
            continue

        rG = grids['r_grid']; zG = grids['z_grid']
        # Valid if r is finite AND above threshold
        valid = np.isfinite(rG) & (rG > r_threshold)

        if not np.any(valid):
            # No islands at this pair
            continue

        labels, nlab = _label_8_connected(valid)

        for lab in range(1, nlab+1):
            mask = (labels == lab)
            size = int(mask.sum())
            if size < min_windows:
                continue

            # Extract stats
            r_vals = rG[mask]
            z_vals = zG[mask]

            # Cluster "mass" above threshold in z-space (strength + extent)
            cluster_mass = float(np.nansum(z_vals - z_thr))

            # Spatial bbox from window rectangles
            x0_min = float(np.nanmin(grids['x0_grid'][mask]))
            x1_max = float(np.nanmax(grids['x1_grid'][mask]))
            y0_min = float(np.nanmin(grids['y0_grid'][mask]))
            y1_max = float(np.nanmax(grids['y1_grid'][mask]))

            # Collect window rects (optional, handy for downstream)
            # (x0,x1,y0,y1) list for all member windows
            # NOTE: if you want to keep indices instead, you can store np.argwhere(mask)
            member_rects = np.column_stack([
                grids['x0_grid'][mask],
                grids['x1_grid'][mask],
                grids['y0_grid'][mask],
                grids['y1_grid'][mask],
            ]).tolist()

            island = {
                'pair': (A, B),
                'label': lab,
                'n_windows': size,
                'median_r': float(np.nanmedian(r_vals)),
                'mean_r': float(np.nanmean(r_vals)),
                'max_r': float(np.nanmax(r_vals)),
                'median_z': float(np.nanmedian(z_vals)),
                'cluster_mass_z': cluster_mass,
                'bbox': (x0_min, x1_max, y0_min, y1_max),
                'window_rects': member_rects,
                'grid_shape': rG.shape,
                'grid_x0s': grids['xs'].tolist(),
                'grid_y0s': grids['ys'].tolist(),
                # Placeholders for later inference / stability:
                'cluster_p': None,      # TODO: fill via permutation-based max-cluster test
                'stability': None       # TODO: fill via bootstrap frequency / IoU
            }
            islands.append(island)
            index_rows.append({
                'class_A': A, 'class_B': B, 'label': lab,
                'n_windows': size,
                'bbox_x0': x0_min, 'bbox_x1': x1_max,
                'bbox_y0': y0_min, 'bbox_y1': y1_max,
                'cluster_mass_z': cluster_mass,
                'median_r': island['median_r']
            })

    # Lightweight index DataFrame for quick filtering/sorting
    island_index = pd.DataFrame(index_rows).sort_values(
        ['class_A','class_B','cluster_mass_z','n_windows'],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)

    return islands, island_index




# %%
islands, island_index = find_islands_for_all_pairs(results_df,
                                                   r_threshold=0.7,
                                                   r_col='weighted_pearson_r',
                                                   min_windows=4)
island_index.head()

# %%
islands[1]['window_rects'][0]
wx1 = islands[1]['window_rects'][0][0]
wx2 = islands[1]['window_rects'][0][1]
wy1 = islands[1]['window_rects'][0][2]
wy2 = islands[1]['window_rects'][0][3]

# %%
allexp[((allexp['class'] == '01 IT-ET Glut') | (allexp['class'] == '03 OB-CR Glut')) & (((allexp['x'] > wx1) & (allexp['x'] < wx2)) & ((allexp['y'] > wy1) & (allexp['y'] < wy2)))]

# %%
import numpy as np
import pandas as pd
from collections import defaultdict

# -------------------------------------------------------------
# Build a fast lookup of island windows per (A,B) as set of (x0,y0)
# -------------------------------------------------------------
def _island_windows_by_pair(islands):
    pair_to_windows = defaultdict(set)
    pair_to_axes = {}
    for isl in islands:
        A, B = isl['pair']
        # store grid axes (assumes consistent grid across islands; OK if repeated)
        pair_to_axes[(A,B)] = (np.array(isl['grid_x0s']), np.array(isl['grid_y0s']))
        for (x0,x1,y0,y1) in isl['window_rects']:
            pair_to_windows[(A,B)].add((float(x0), float(y0)))
    return pair_to_windows, pair_to_axes

# -------------------------------------------------------------
# Extract the global window grid (xs, ys) and win_size from results_df
# (assumes a consistent grid used when computing r maps)
# -------------------------------------------------------------
def _extract_grid_from_results(results_df):
    xs = np.array(sorted(results_df['x0'].unique()), dtype=float)
    ys = np.array(sorted(results_df['y0'].unique()), dtype=float)
    # infer win_size from first row
    r0 = results_df.iloc[0]
    win_size_x = float(r0['x1'] - r0['x0'])
    win_size_y = float(r0['y1'] - r0['y0'])
    assert np.isclose(win_size_x, win_size_y), "Non-square windows not supported here."
    return xs, ys, win_size_x

# -------------------------------------------------------------
# Find all window (x0,y0) that cover a given point (x,y)
# -------------------------------------------------------------
def _covering_windows(x, y, xs, ys, win_size):
    # windows with x0 <= x < x0+win_size and y0 <= y < y0+win_size
    x_mask = (xs <= x) & (x < xs + win_size)
    y_mask = (ys <= y) & (y < ys + win_size)
    xi = np.where(x_mask)[0]
    yi = np.where(y_mask)[0]
    # Cartesian product of indices
    return [(float(xs[j]), float(ys[i])) for i in yi for j in xi]

# -------------------------------------------------------------
# Main: per-cell binary encoding + auxiliary table with coverage
# -------------------------------------------------------------
def encode_cell_colocalization(allexp, results_df, islands, theta=0.5, k=3, cell_colname='class'):
    """
    For each cell of type A:
      - collect all sliding windows covering the cell (support)
      - for each partner B != A: compute coverage = (#covering windows that are island windows for (A,B)) / (#covering windows)
      - set binary 1 if coverage >= theta and support >= k; else 0
    Returns:
      binary_mat: DataFrame [n_cells x n_types]
      aux:        DataFrame with per-cell diagnostics (support, per-partner coverage)
    """
    xs, ys, win_size = _extract_grid_from_results(results_df)
    pair_to_windows, _ = _island_windows_by_pair(islands)

    cell_types = sorted(allexp[cell_colname].unique())
    n = len(allexp)

    # Prepare outputs
    bin_data = {t: np.zeros(n, dtype=int) for t in cell_types}  # self-col will stay 0
    aux_rows = []

    # Precompute for speed: all covering windows per cell
    # (keeps exact geometry; typical stride = win_size/2 => up to 4 windows per cell)
    all_cover = []
    for idx, row in allexp[['x','y']].iterrows():
        cov = _covering_windows(float(row['x']), float(row['y']), xs, ys, win_size)
        all_cover.append(cov)

    # Compute encoding
    for idx, row in allexp.iterrows():
        A = row[cell_colname]
        covered = all_cover[idx]
        support = len(covered)

        # Per-partner coverage tracker
        cov_map = {}

        if support >= k:
            covered_set = set(covered)
            for B in cell_types:
                if B == A:
                    cov_map[B] = 0.0
                    continue
                # Island windows for pair (A,B) (order-agnostic lookup)
                key = (A,B) if (A,B) in pair_to_windows else (B,A)
                if key not in pair_to_windows:
                    cov_map[B] = 0.0
                    continue
                island_windows = pair_to_windows[key]
                hit = len(covered_set & island_windows)
                coverage = hit / support if support > 0 else 0.0
                cov_map[B] = coverage
                # binary decision
                if coverage >= theta:
                    bin_data[B][idx] = 1
        else:
            # insufficient support: keep zeros, record coverage as 0 for all partners
            for B in cell_types:
                cov_map[B] = 0.0

        # Build aux row
        aux_row = {
            'cell_index': idx,
            'x': float(row['x']),
            'y': float(row['y']),
            'cell_type': A,
            'support_windows': support
        }
        # add per-partner coverages (e.g., coverage_B)
        for B in cell_types:
            if B == A:
                aux_row[f'coverage_{B}'] = 0.0
            else:
                aux_row[f'coverage_{B}'] = float(cov_map[B])
        aux_rows.append(aux_row)

    binary_mat = pd.DataFrame(bin_data, index=allexp.index)
    # enforce self-col = 0 (safety)
    for A in cell_types:
        binary_mat.loc[allexp[cell_colname] == A, A] = 0

    aux = pd.DataFrame(aux_rows).set_index('cell_index').loc[allexp.index]

    return binary_mat, aux


# %%
# Example:
binary_mat, aux = encode_cell_colocalization(allexp, results_df, islands, theta=0.5, k=3)

# %%
binary_mat['cell_label'] = allexp['cell_label'].values
aux['cell_label'] = allexp['cell_label'].values
# make cell_label index
binary_mat = binary_mat.set_index('cell_label')
aux = aux.set_index('cell_label')

# %%
allexp.shape

# %%
# keep allexp columns from 22 to end except last 50
allexp_sub = allexp.iloc[:, 22:-50]

allexp_sub.shape

# %% [markdown]
# # Feature Matrix Formation

# %% [markdown]
# ## Ligand-Receptor Genes Calculation

# %%
LR_pairs = pd.read_csv('../data/mouse_850_lr_pairs_cpdb_interactions.csv')
LR_pairs.shape

# %%
import pandas as pd
import numpy as np

def prepare_lr_features(
    allexp: pd.DataFrame,
    lr_pairs: pd.DataFrame,
    ligand_col: str = "ligand_genesymbol",   
    receptor_col: str = "target_genesymbol", 
    meta_cols = ("x","y","class")            
):
    """
    Build per-cell receptor and ligand expression matrices for exactly the LR pairs provided.

    Inputs
    ------
    allexp : DataFrame
        Rows = cells (index = your cell ids/labels), columns include gene expression + meta columns.
    lr_pairs : DataFrame
        Must contain columns with ligand and receptor gene symbols (one row per LR pair).
    ligand_col, receptor_col : str
        Column names in lr_pairs for ligand and receptor symbols.
    meta_cols : iterable
        Columns in allexp that are NOT genes (will be excluded).

    Outputs
    -------
    X_receptors : DataFrame  (cells × unique_receptors_kept)
    X_ligands   : DataFrame  (cells × unique_ligands_kept)
    lr_pairs_kept : DataFrame (filtered to pairs present in allexp, with integer columns
                     'ligand_idx' and 'receptor_idx' giving column positions in X_ligands/X_receptors)
    report : dict  (counts and lists of dropped/mapped genes)
    """
    # --- sanitize / standardize gene symbols (upper-case) to improve matching ---
    def _upper_series(s):
        return s.astype(str).str.strip().str.upper()

    # Make a copy and upper-case gene names in allexp columns (genes only)
    allexp_cols = pd.Index(allexp.columns)
    meta_cols = [c for c in meta_cols if c in allexp_cols]
    gene_cols = [c for c in allexp_cols if c not in meta_cols]

    # Build a mapping original->UPPER for allexp gene columns
    gene_cols_upper = pd.Index([str(c).upper() for c in gene_cols])
    colmap = dict(zip(gene_cols_upper, gene_cols))  # UPPER -> original

    # Upper-case ligand/receptor symbols
    lig_syms = _upper_series(lr_pairs[ligand_col])
    rec_syms = _upper_series(lr_pairs[receptor_col])

    # Compose a filtered LR table with upper-cased symbols
    lr_uc = lr_pairs.copy()
    lr_uc["_LIG"] = lig_syms
    lr_uc["_REC"] = rec_syms

    # --- keep only pairs whose both genes are present in allexp ---
    # present_lig = gene_cols_upper.isin(lr_uc["_LIG"]).to_numpy()
    # present_rec = gene_cols_upper.isin(lr_uc["_REC"]).to_numpy()
    genes_in_allexp_uc = set(gene_cols_upper)

    keep_mask = lr_uc["_LIG"].isin(genes_in_allexp_uc) & lr_uc["_REC"].isin(genes_in_allexp_uc)
    lr_pairs_kept = lr_uc.loc[keep_mask].reset_index(drop=True)

    # Unique ligands/receptors actually present
    uniq_lig_uc = list(dict.fromkeys(lr_pairs_kept["_LIG"]))  # preserve order of first appearance
    uniq_rec_uc = list(dict.fromkeys(lr_pairs_kept["_REC"]))

    # Map back to original column names in allexp
    uniq_lig_cols = [colmap[g] for g in uniq_lig_uc]
    uniq_rec_cols = [colmap[g] for g in uniq_rec_uc]

    # --- slice allexp into ligand/receptor matrices (cells × genes) ---
    X_ligands = allexp.loc[:, uniq_lig_cols].copy()
    X_receptors = allexp.loc[:, uniq_rec_cols].copy()

    # Optional: ensure numeric dtype
    X_ligands = X_ligands.apply(pd.to_numeric, errors="coerce")
    X_receptors = X_receptors.apply(pd.to_numeric, errors="coerce")

    # --- annotate lr_pairs_kept with column indices into X_ligands / X_receptors ---
    lig_idx_map = {g: i for i, g in enumerate(uniq_lig_cols)}
    rec_idx_map = {g: i for i, g in enumerate(uniq_rec_cols)}

    lr_pairs_kept["ligand_symbol_uc"] = lr_pairs_kept["_LIG"]
    lr_pairs_kept["receptor_symbol_uc"] = lr_pairs_kept["_REC"]
    lr_pairs_kept["ligand_symbol"] = lr_pairs_kept["_LIG"].map(colmap)
    lr_pairs_kept["receptor_symbol"] = lr_pairs_kept["_REC"].map(colmap)
    lr_pairs_kept["ligand_idx"] = lr_pairs_kept["ligand_symbol"].map(lig_idx_map)
    lr_pairs_kept["receptor_idx"] = lr_pairs_kept["receptor_symbol"].map(rec_idx_map)

    # --- reporting ---
    dropped_pairs = lr_uc.loc[~keep_mask, [ligand_col, receptor_col]]
    missing_ligs = sorted(set(lr_uc["_LIG"]) - genes_in_allexp_uc)
    missing_recs = sorted(set(lr_uc["_REC"]) - genes_in_allexp_uc)

    report = {
        "n_pairs_input": int(len(lr_pairs)),
        "n_pairs_kept": int(len(lr_pairs_kept)),
        "n_unique_ligands_kept": int(len(uniq_lig_cols)),
        "n_unique_receptors_kept": int(len(uniq_rec_cols)),
        "missing_ligands_from_allexp": [colmap.get(g, g) for g in missing_ligs],  # best effort
        "missing_receptors_from_allexp": [colmap.get(g, g) for g in missing_recs],
        "dropped_pairs": dropped_pairs
    }

    # Clean columns for return
    lr_pairs_kept = lr_pairs_kept.drop(columns=["_LIG","_REC"])

    return X_receptors, X_ligands, lr_pairs_kept, report


# %%
# ---------------------------
# Example usage:
X_receptors, X_ligands, lr_pairs_kept, report = prepare_lr_features(allexp, LR_pairs,
    ligand_col="ligand_genesymbol", receptor_col="target_genesymbol", meta_cols=[])
# ---------------------------

# %% [markdown]
# ## Ligand Exposure Calculation

# %%
# ---- Reuse your helpers (already defined above) -----------------------------
# _extract_grid_from_results, _island_windows_by_pair, _covering_windows

def _preindex_window_cells(allexp, xs, ys, win_size, class_col="class"):
    """
    Pre-index cells by (x0,y0) window and by cell type for fast neighbor lookup.
    Returns: dict[(x0,y0)] -> dict[class_name] -> np.array(cell_indices)
    """
    x = allexp["x"].to_numpy(float)
    y = allexp["y"].to_numpy(float)
    classes = allexp[class_col].astype(str).to_numpy()

    # For each window start (x0,y0), build a boolean mask of cells inside it
    win_index = {}
    for x0 in xs:
        x1 = x0 + win_size
        x_mask = (x >= x0) & (x < x1)
        for y0 in ys:
            y1 = y0 + win_size
            y_mask = (y >= y0) & (y < y1)
            idx = np.where(x_mask & y_mask)[0]
            if idx.size == 0:
                continue
            # bucket by class for this window
            by_class = {}
            for c in np.unique(classes[idx]):
                by_class[c] = idx[classes[idx] == c]
            win_index[(float(x0), float(y0))] = by_class
    return win_index


def compute_ligand_exposure(
    allexp: pd.DataFrame,
    X_ligands: pd.DataFrame,
    results_df: pd.DataFrame,
    islands: list,
    aux: pd.DataFrame,
    theta: float = 0.5,              # coverage threshold to deem A↔B island-sharing
    k_support: int = 3,              # min #covering windows for the receiver cell
    mode: str = "mean",              # "mean" or "kde"
    sigma: float | None = None,      # KDE sigma in coordinate units; default = win_size/3
    class_col: str = "class"
) -> pd.DataFrame:
    """
    Step 2 — Compute ligand exposure via colocalization islands.

    For each receiver cell i:
      1) Find partner types B with aux.loc[i, f"coverage_{B}"] >= theta AND support_windows >= k_support
      2) Collect neighbors = union of cells of those B-types that fall in ANY sliding window covering i
         AND that window is an island window for the (A_i, B) pair.
      3) Exposure for each ligand gene g = average (mode="mean") or KDE-weighted average (mode="kde")
         of X_ligands[g] across the neighbor cells.

    Returns:
      X_exposure: DataFrame (cells × ligand genes), index aligned with allexp / X_ligands
    """
    # --- grid and island caches ---
    xs, ys, win_size = _extract_grid_from_results(results_df)
    pair_to_windows, _ = _island_windows_by_pair(islands)  # {(A,B): set[(x0,y0)], ...}

    # pre-index cells per (x0,y0) window and class for quick union queries
    win_index = _preindex_window_cells(allexp, xs, ys, win_size, class_col=class_col)

    # map from aux columns "coverage_B" → B
    coverage_cols = {
        c.replace("coverage_", ""): c
        for c in aux.columns if c.startswith("coverage_")
    }

    # output matrix
    X_exposure = pd.DataFrame(0.0, index=allexp.index, columns=X_ligands.columns)

    # coords & class arrays for KDE mode
    coord = allexp[["x", "y"]].to_numpy(float)
    classes = allexp[class_col].astype(str).to_numpy()

    # default sigma ~ window size / 3 (smooth inside an island window)
    if sigma is None:
        sigma = win_size / 3.0 if np.isfinite(win_size) and win_size > 0 else 1.0
    inv2sig2 = 1.0 / (2.0 * (sigma ** 2))

    # --- main loop over receiver cells ---
    for i in range(len(allexp)):
        A = classes[i]
        # support (how many windows cover this cell)
        covered_i = _covering_windows(coord[i,0], coord[i,1], xs, ys, win_size)
        support = len(covered_i)
        if support < k_support:
            continue  # exposure stays 0

        # partner types B that share island with i (per aux coverage threshold)
        eligible_B = [B for B, ccol in coverage_cols.items()
                      if B != A and aux.iloc[i][ccol] >= theta]

        if not eligible_B:
            continue

        # collect neighbors: union over B and over windows that cover i,
        # intersected with island windows for (A,B)
        nbr_idx = set()
        covered_set = set(covered_i)
        for B in eligible_B:
            key = (A, B) if (A, B) in pair_to_windows else (B, A)
            if key not in pair_to_windows:
                continue
            # windows that both cover i AND are island windows for (A,B)
            isl_windows = pair_to_windows[key] & covered_set
            if not isl_windows:
                continue
            # gather all B-type cells inside those windows
            for w in isl_windows:
                by_class = win_index.get(w)
                if not by_class:
                    continue
                idxB = by_class.get(B)
                if idxB is not None and idxB.size:
                    nbr_idx.update(idxB.tolist())

        if not nbr_idx:
            continue

        nbr_idx = np.fromiter(nbr_idx, dtype=int, count=len(nbr_idx))

        if mode == "mean":
            # simple average across eligible neighbors
            X_exposure.iloc[i, :] = X_ligands.iloc[nbr_idx, :].mean(axis=0).fillna(0.0).to_numpy()

        elif mode == "kde":
            # Gaussian weights by distance to receiver cell i (within chosen sigma)
            d2 = np.sum((coord[nbr_idx] - coord[i])**2, axis=1)  # squared distances
            w = np.exp(-d2 * inv2sig2)
            w_sum = np.sum(w)
            if w_sum <= 0:
                continue
            w = w / w_sum
            # weighted average per ligand gene
            # (vectorized: neighbors × genes  @ weights)
            X_exposure.iloc[i, :] = np.dot(w, X_ligands.iloc[nbr_idx, :].to_numpy())

        else:
            raise ValueError("mode must be 'mean' or 'kde'")

    return X_exposure


# %%
# Pick a threshold consistent with your binary encoding step
theta = 0.5
k_support = 3

# Mean-based exposure (fast, good baseline)
X_exposure_mean = compute_ligand_exposure(
    allexp=allexp,
    X_ligands=X_ligands,
    results_df=results_df,     # from your (weighted) PCC stage
    islands=islands,           # from find_islands_for_all_pairs(...)
    aux=aux,                   # has coverage_* and support_windows
    theta=theta,
    k_support=k_support,
    mode="mean"
)

# KDE-weighted exposure (distance-weighted inside island windows)
X_exposure_kde = compute_ligand_exposure(
    allexp=allexp,
    X_ligands=X_ligands,
    results_df=results_df,
    islands=islands,
    aux=aux,
    theta=theta,
    k_support=k_support,
    mode="kde",
    sigma=None   # defaults to win_size/3
)

# Quick QC:
print("Nonzero exposure frac (mean):",
      (X_exposure_mean.values > 0).mean())
print("Nonzero exposure frac (kde):",
      (X_exposure_kde.values > 0).mean())


# %%
X_exposure_kde

# %% [markdown]
# ## LR Interaction Scoring

# %%
import numpy as np
import pandas as pd

def build_lr_interaction_features(
    X_receptors: pd.DataFrame,
    X_exposure: pd.DataFrame,          # ligand *exposure* matrix (not raw ligand expr)
    lr_pairs_kept: pd.DataFrame,       # from Step 1 (already filtered to present genes)
    ligand_col: str = "ligand_symbol",
    receptor_col: str = "receptor_symbol",
    method: str = "product",           # "product" or "min"
    suffix: str = ""                   # optional suffix for column names, e.g., "_prod"
):
    """
    Step 3 — Build LR interaction features.

    Inputs
    ------
    X_receptors : DataFrame (cells × unique receptor genes)
    X_exposure  : DataFrame (cells × unique ligand genes)  [ligand *exposure*]
    lr_pairs_kept : DataFrame containing at least [ligand_col, receptor_col]
                    and (optionally) integer columns 'ligand_idx','receptor_idx'
                    that index into X_exposure / X_receptors respectively.
    method : "product" (R * Lexp) or "min" (min(R, Lexp))
    suffix : optional string appended to interaction column names.

    Returns
    -------
    X_LR  : DataFrame (cells × n_pairs), interaction per LR pair
    X_aux : DataFrame with receptor-only and ligand-exposure-only cols used
    meta  : dict with bookkeeping (pair->indices, method)
    """
    # --- Resolve indices for each pair into the receptor/exposure matrices ---
    # Prefer the precomputed indices from Step 1 if available (fast, robust).
    have_idx = {"ligand_idx" in lr_pairs_kept.columns,
                "receptor_idx" in lr_pairs_kept.columns}
    have_idx = all(have_idx)

    # Maps for name→position (fallback if indices absent)
    rec_pos = {g: i for i, g in enumerate(X_receptors.columns)}
    lig_pos = {g: i for i, g in enumerate(X_exposure.columns)}

    # Build ordered lists of positions and names aligned to lr_pairs_kept rows
    lig_names = []
    rec_names = []
    lig_idx = []
    rec_idx = []

    for _, row in lr_pairs_kept.iterrows():
        L = row[ligand_col]
        R = row[receptor_col]

        # If indices were carried from Step 1, use them; else resolve by column name.
        if have_idx:
            li = int(row["ligand_idx"])
            ri = int(row["receptor_idx"])
            # sanity: ensure columns still match names
            assert X_exposure.columns[li] == L, f"Ligand index/name mismatch: {L}"
            assert X_receptors.columns[ri] == R, f"Receptor index/name mismatch: {R}"
        else:
            # Fallback resolution via column names
            if L not in lig_pos or R not in rec_pos:
                # Skip pairs whose genes are missing (should be rare after Step 1 filtering)
                continue
            li = lig_pos[L]
            ri = rec_pos[R]

        lig_names.append(L)
        rec_names.append(R)
        lig_idx.append(li)
        rec_idx.append(ri)

    n_pairs = len(lig_idx)
    if n_pairs == 0:
        raise ValueError("No LR pairs could be aligned to X_receptors/X_exposure.")

    # --- Pull the arrays we need (vectorized over cells × pairs) ---
    # R: receptor expression for each pair
    R = X_receptors.iloc[:, rec_idx].to_numpy(dtype=float)   # shape: (cells, n_pairs)
    # Lexp: ligand *exposure* for each pair
    Lexp = X_exposure.iloc[:, lig_idx].to_numpy(dtype=float) # shape: (cells, n_pairs)

    # --- Interaction function ---
    if method.lower() == "product":
        S = R * Lexp
        method_used = "product"
        suf = suffix or "_prod"
    elif method.lower() == "min":
        S = np.minimum(R, Lexp)
        method_used = "min"
        suf = suffix or "_min"
    else:
        raise ValueError("method must be 'product' or 'min'")

    # --- Build nice column names like 'LIGAND|RECEPTOR_prod' ---
    lr_labels = [f"{L}|{R}{suf}" for L, R in zip(lig_names, rec_names)]
    X_LR = pd.DataFrame(S, index=X_receptors.index, columns=lr_labels)

    # --- Auxiliary matrices used in these interactions (optional but useful) ---
    # receptor-only features (just those receptors that appear in pairs, once)
    # ligand-only (exposure) features (just those ligands that appear in pairs, once)
    # We keep column order matched to the *first* time a gene appears in lr_pairs_kept.
    uniq_rec_ordered = list(dict.fromkeys(rec_names))
    uniq_lig_ordered = list(dict.fromkeys(lig_names))

    X_rec_used  = X_receptors.loc[:, uniq_rec_ordered].copy()
    X_lig_used  = X_exposure.loc[:, uniq_lig_ordered].copy()
    # Prefix to make it explicit in modeling design matrices
    X_rec_used.columns = [f"R__{c}"     for c in X_rec_used.columns]
    X_lig_used.columns = [f"Lexp__{c}"  for c in X_lig_used.columns]

    X_aux = pd.concat([X_rec_used, X_lig_used], axis=1)

    meta = {
        "method": method_used,
        "pairs": list(zip(lig_names, rec_names)),
        "ligand_indices": lig_idx,
        "receptor_indices": rec_idx,
        "n_pairs": n_pairs
    }
    return X_LR, X_aux, meta


# %%
# ---------------------------
# Example usage
# ---------------------------
X_LR_prod, X_aux, meta = build_lr_interaction_features(
    X_receptors=X_receptors,
    X_exposure=X_exposure_mean,         # or X_exposure_kde
    lr_pairs_kept=lr_pairs_kept,
    ligand_col="ligand_symbol",
    receptor_col="receptor_symbol",
    method="product"
)

# X_LR_min, X_aux_min, meta_min = build_lr_interaction_features(..., method="min")

# Quick sanity checks:
# 1) No NaNs, non-negativity if inputs are non-negative
assert np.isfinite(X_LR_prod.to_numpy()).all()
# 2) Column count equals number of kept pairs
assert X_LR_prod.shape[1] == meta["n_pairs"]

# %% [markdown]
# ## Gene Filtering
# 
# Remove ligands, receptors and bottom 5% genes in terms of variance

# %%
ligands = LR_pairs['ligand_genesymbol'].unique()
receptors = LR_pairs['target_genesymbol'].unique()

# %%
aux = aux.reset_index().rename(columns={'index': 'cell_label'})

# %%
# You should already have these from Steps 1–3 & island encoding:
# - allexp: DataFrame (cells × [meta + genes]); index = cell ids; columns include 'x','y','class','cell_label' etc.
# - X_receptors: DataFrame (cells × receptors)              # Step 1
# - X_exposure:  DataFrame (cells × ligands)                # Step 2
# - X_LR:        DataFrame (cells × LR-pair interactions)   # Step 3
# - aux:         DataFrame with 'support_windows' and columns starting with 'coverage_'
# - lr_pairs_kept: DataFrame with 'ligand_symbol', 'receptor_symbol'

# Sanity:
for name in ["allexp", "X_receptors", "X_exposure_kde", "X_LR_prod", "aux", "lr_pairs_kept"]:
    assert name in globals(), f"Missing variable: {name}"

# Align row order across all matrices by allexp:
idx = allexp.index
X_receptors = X_receptors.loc[idx]
X_exposure  = X_exposure_kde.loc[idx]
X_LR        = X_LR_prod.loc[idx]
aux         = aux.loc[idx]



# CONFIG
META_COLS          = ["x","y","class","cell_label"]  # adjust if you have different meta cols
DROP_LIGANDS       = True     # recommended to avoid leakage; set False if you want to include ligands as targets
MIN_DETECT_FRAC    = 0.01     # drop targets detected in <1% of cells
MIN_VAR_QUANT      = 0.05     # drop bottom 5% variance targets
DROP_TECHNICALS    = True     # drop mito/ribo/hb-like genes if present
COV_PREFIX         = "coverage_"

def get_gene_matrix(allexp: pd.DataFrame, meta_cols=META_COLS) -> pd.DataFrame:
    """Return cells × genes numeric matrix by dropping meta columns."""
    meta_cols = [c for c in meta_cols if c in allexp.columns]
    gene_cols = [c for c in allexp.columns if c not in meta_cols]
    expr = allexp.loc[:, gene_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return expr

def zscore_df(df: pd.DataFrame):
    """Z-score columns; returns (Z, scaler)."""
    sc = StandardScaler(with_mean=True, with_std=True)
    Z = sc.fit_transform(df.values)
    return pd.DataFrame(Z, index=df.index, columns=df.columns), sc


# %%
# 1) Full cells×genes matrix
expr_all = get_gene_matrix(allexp_sub, meta_cols=META_COLS)

# 2) Receptor & ligand sets from your LR list
receptors = sorted(set(lr_pairs_kept["receptor_symbol"]))
ligands   = sorted(set(lr_pairs_kept["ligand_symbol"]))

# 3) Build the drop list for targets
drop = set(expr_all.columns).intersection(receptors)
if DROP_LIGANDS:
    drop |= set(expr_all.columns).intersection(ligands)

# 4) Filter low detection / low variance
det_frac = (expr_all > 0).mean(axis=0)                            # fraction of cells with nonzero expression
var_g    = expr_all.var(axis=0)
low_det  = det_frac[det_frac < MIN_DETECT_FRAC].index
low_var  = var_g[var_g < var_g.quantile(MIN_VAR_QUANT)].index
drop |= set(low_det) | set(low_var)

if DROP_TECHNICALS:
    tech_re = re.compile(r"^(MT-|mt-|RPL|RPS|HBA|HBB)")
    tech = [g for g in expr_all.columns if tech_re.match(g)]
    drop |= set(tech)

# 6) Final targets
target_genes = [g for g in expr_all.columns if g not in drop]
Y_targets = expr_all.loc[idx, target_genes].copy()

print(f"Targets kept: {Y_targets.shape[1]} genes (from {expr_all.shape[1]} total).")


# %%
# Use every coverage_* column (broader approach)
coverage_cols_all = [c for c in aux.columns if c.startswith(COV_PREFIX)]
assert len(coverage_cols_all) > 0, "No coverage_* columns in aux."

X_cov = aux[coverage_cols_all].copy()
# Optional rename to cleaner names for modeling; keep originals if you prefer
X_cov.columns = [f"cov::{c.replace(COV_PREFIX,'')}" for c in coverage_cols_all]

# Sample weights from window support (use later in training)
sample_weight = aux["support_windows"].clip(lower=1).to_numpy()

# %%
# Z-score each block independently (keep scalers to transform CV folds later)
X_receptors_z, sc_R = zscore_df(X_receptors)
X_exposure_z,  sc_E = zscore_df(X_exposure)
X_LR_z,        sc_I = zscore_df(X_LR)
X_cov_z,       sc_C = zscore_df(X_cov)

# Concatenate in fixed order
X = pd.concat([X_receptors_z, X_exposure_z, X_LR_z, X_cov_z], axis=1)

print("X shape:", X.shape)                # (cells × features)
print("Y_targets shape:", Y_targets.shape) # (cells × target genes)


# %%
# Basic sanity
import numpy as np
assert np.isfinite(X.values).all(), "Non-finite values in X"
assert np.isfinite(Y_targets.values).all(), "Non-finite values in Y_targets"

# Sparsity snapshots (pre-zscore)
print("Nonzero frac (receptors):", (X_receptors.values != 0).mean())
print("Nonzero frac (exposure) :", (X_exposure.values  != 0).mean())
print("Nonzero frac (LR)       :", (X_LR.values        != 0).mean())
print("Mean coverage (X_cov)   :", X_cov.mean().mean())

# Ready for Step 5: spatial block CV + multi-task Elastic Net / RF

# ---- Repro & CV/Test config ----
SEED = 42
N_GROUPS = 10          # number of spatial groups to form from (x,y)
TEST_FRACTION = 0.2    # ~20% groups as final test set
N_SPLITS = 5           # GroupKFold folds on the dev set

# ---- Elastic Net search grid ----
ALPHAS = np.logspace(-4, 1, 8)      # 1e-4 ... 10
L1S    = [0.1, 0.5, 0.9]            # l1_ratio

# ---- Metric aggregation choice ----
AGG = "mean"   # "mean" or "median" R^2 across targets
USE_WEIGHTS_IN_SCORING = True  # use support_windows as weights in metrics


# %%
# (x,y) → spatial groups
coords = allexp[["x","y"]].to_numpy(dtype=float)

km = KMeans(n_clusters=N_GROUPS, n_init=10, random_state=SEED)
group_labels = km.fit_predict(coords)  # 0..N_GROUPS-1, one label per cell

# Decide test groups (≈ TEST_FRACTION of groups)
rng = np.random.default_rng(SEED)
unique_groups = np.arange(N_GROUPS)
n_test_groups = max(1, int(round(TEST_FRACTION * N_GROUPS)))
test_groups = rng.choice(unique_groups, size=n_test_groups, replace=False)

is_test = np.isin(group_labels, test_groups)
is_dev  = ~is_test

print("Groups:", N_GROUPS, "| Test groups:", sorted(test_groups.tolist()))
print("Dev cells:", is_dev.sum(), " Test cells:", is_test.sum())

# %%
def fit_transform_split(X, Y, train_idx, val_idx):
    """
    Fit scalers on TRAIN only; transform train/val for both X and Y.
    Returns: Xtr, Xva, Ytr, Yva, scalers (sx, sy)
    """
    sx = StandardScaler(with_mean=True, with_std=True)
    sy = StandardScaler(with_mean=True, with_std=True)

    Xtr = sx.fit_transform(X[train_idx])
    Xva = sx.transform(X[val_idx])

    Ytr = sy.fit_transform(Y[train_idx])
    Yva = sy.transform(Y[val_idx])

    return Xtr, Xva, Ytr, Yva, sx, sy

def weighted_r2_per_target(y_true, y_pred, sample_weight=None):
    """
    R² per target column (multioutput). sklearn's r2_score with multioutput=None
    for each column; supports sample weights if provided.
    """
    T = y_true.shape[1]
    r2s = np.empty(T, dtype=float)
    for t in range(T):
        r2s[t] = r2_score(y_true[:, t], y_pred[:, t], sample_weight=sample_weight)
    return r2s

def aggregate_scores(r2_vec, agg="mean"):
    return float(np.nanmean(r2_vec)) if agg == "mean" else float(np.nanmedian(r2_vec))


# %%
# Slice dev/test once
X_all = X.to_numpy(dtype=float)
Y_all = Y_targets.to_numpy(dtype=float)

groups_dev = group_labels[is_dev]
X_dev, Y_dev = X_all[is_dev], Y_all[is_dev]
weights_all = aux["support_windows"].to_numpy()
w_dev = weights_all[is_dev] if USE_WEIGHTS_IN_SCORING else None

gkf = GroupKFold(n_splits=N_SPLITS)

cv_results = []  # collect dicts for a summary table

for alpha, l1 in product(ALPHAS, L1S):
    fold_scores = []
    per_target_scores = []  # optional: store mean per-target R² across folds too

    for tr_idx, va_idx in gkf.split(X_dev, groups=groups_dev):
        # Train/val split indexes relative to DEV subset
        Xtr, Xva, Ytr, Yva, sx, sy = fit_transform_split(X_dev, Y_dev, tr_idx, va_idx)

        # Model
        model = MultiTaskElasticNet(
            alpha=alpha,
            l1_ratio=l1,
            fit_intercept=False,   # we already standardized
            max_iter=5000,
            random_state=SEED,
            selection="cyclic"
        )
        model.fit(Xtr, Ytr)

        # Predict and score
        Yhat = model.predict(Xva)
        sw = w_dev[va_idx] if w_dev is not None else None
        r2_t = weighted_r2_per_target(Yva, Yhat, sample_weight=sw)
        fold_scores.append(aggregate_scores(r2_t, AGG))
        per_target_scores.append(r2_t)

    cv_results.append({
        "alpha": alpha,
        "l1_ratio": l1,
        "cv_score": float(np.mean(fold_scores)),
        "cv_score_std": float(np.std(fold_scores)),
        "per_target_mean": float(np.mean(np.vstack(per_target_scores), axis=0).mean()),
    })

# Pick best by cv_score
cv_df = pd.DataFrame(cv_results).sort_values(["cv_score", "per_target_mean"], ascending=[False, False]).reset_index(drop=True)
best = cv_df.iloc[0].to_dict()
best_alpha, best_l1 = float(best["alpha"]), float(best["l1_ratio"])

print("Best hyperparams → alpha=%.4g, l1_ratio=%.2f | CV mean %s R²=%.4f (± %.4f)" %
      (best_alpha, best_l1, AGG, best["cv_score"], best["cv_score_std"]))

cv_df.head(10)
