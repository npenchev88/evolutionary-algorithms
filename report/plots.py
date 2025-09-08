# plots.py
import os; os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.lines import Line2D

def _norm_method(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.upper().str.replace(r"\s+", "", regex=True))

def _quantize(x, eps):
    x = np.asarray(x, float)
    return np.round(x / eps) * eps

def plot_pareto(df_fronts, N, out_path, eps=1e-6, cap_random=None):
    """Risk (f2) vs Value (-f1) with explicit overlap handling:
       RANDOM = triangles, NSGA2 = circles;
       overlapping points are drawn as both markers with tiny symmetric offsets.
    """
    plt.figure(figsize=(10, 8))

    df = df_fronts.copy()
    df = df[df["N"] == N]
    if df.empty:
        plt.title(f"Pareto Fronts for N={N} (No data)")
        plt.savefig(out_path, dpi=150, bbox_inches="tight"); plt.close(); return

    # normalize + ensure numeric
    df["method"] = _norm_method(df["method"])
    df["f1"] = pd.to_numeric(df["f1"], errors="coerce")
    df["f2"] = pd.to_numeric(df["f2"], errors="coerce")
    df = df.dropna(subset=["f1","f2"])
    df["value"] = -df["f1"]
    df["risk"]  =  df["f2"]

    nsga = df[df["method"] == "NSGA2"].copy()
    rand = df[df["method"] == "RANDOM"].copy()

    if cap_random is not None and len(rand) > cap_random:
        rand = rand.sample(cap_random, random_state=0)

    # ----- overlap detection via key-set (no fragile indices) -----
    if not nsga.empty and not rand.empty:
        nsga["kr"] = _quantize(nsga["risk"],  eps)
        nsga["kv"] = _quantize(nsga["value"], eps)
        rand["kr"] = _quantize(rand["risk"],  eps)
        rand["kv"] = _quantize(rand["value"], eps)

        key_tuples_nsga = list(map(tuple, nsga[["kr","kv"]].to_numpy()))
        key_tuples_rand = list(map(tuple, rand[["kr","kv"]].to_numpy()))
        overlap_keys = set(key_tuples_nsga).intersection(key_tuples_rand)

        nsga_overlap_mask = [k in overlap_keys for k in key_tuples_nsga]
        rand_overlap_mask = [k in overlap_keys for k in key_tuples_rand]

        nsga_both = nsga.loc[nsga_overlap_mask].drop(columns=["kr","kv"])
        rand_both = rand.loc[rand_overlap_mask].drop(columns=["kr","kv"])
        nsga_only = nsga.loc[[not m for m in nsga_overlap_mask]].drop(columns=["kr","kv"])
        rand_only = rand.loc[[not m for m in rand_overlap_mask]].drop(columns=["kr","kv"])
    else:
        nsga_only, rand_only = nsga, rand
        nsga_both = nsga.iloc[0:0].copy()
        rand_both = rand.iloc[0:0].copy()

    # tiny symmetric offsets to make overlaps visible
    rx, ry = df["risk"], df["value"]
    dx = 0.005 * (rx.max() - rx.min() + 1e-12)
    dy = 0.005 * (ry.max() - ry.min() + 1e-12)

    # draw: RANDOM under, NSGA2 above; overlaps as offset pair
    if not rand_only.empty:
        plt.scatter(rand_only["risk"], rand_only["value"],
                    marker="^", s=24, alpha=0.6, label="RANDOM",
                    edgecolors="none", zorder=2)
    if not nsga_only.empty:
        plt.scatter(nsga_only["risk"], nsga_only["value"],
                    marker="o", s=30, alpha=0.9, label="NSGA2",
                    edgecolors="none", zorder=3)

    if not nsga_both.empty:
        plt.scatter(rand_both["risk"] - dx, rand_both["value"] - dy,
                    marker="^", s=28, alpha=0.7, edgecolors="none",
                    color="#ff7f0e", zorder=4)
        plt.scatter(nsga_both["risk"] + dx, nsga_both["value"] + dy,
                    marker="o", s=34, alpha=0.95, edgecolors="k",
                    facecolors="#1f77b4", linewidths=0.4, zorder=5)

    # faint convex hulls (optional)
    for sub in (rand_only, nsga_only):
        if len(sub) > 2:
            try:
                pts = np.c_[sub["risk"].to_numpy(), sub["value"].to_numpy()]
                hull = ConvexHull(pts)
                for i, j in hull.simplices:
                    plt.plot(pts[[i, j], 0], pts[[i, j], 1],
                             "k-", alpha=0.12, linewidth=1, zorder=1)
            except Exception:
                pass

    # ensure legend shows both labels even if one set happens to be empty
    handles, labels = plt.gca().get_legend_handles_labels()
    needed = {"NSGA2": ("o", "#1f77b4"), "RANDOM": ("^", "#ff7f0e")}
    have = set(labels)
    for name, (mk, col) in needed.items():
        if name not in have:
            handles.append(Line2D([0],[0], marker=mk, color="w",
                                  markerfacecolor=col,
                                  markeredgecolor="k" if name=="NSGA2" else "none",
                                  markersize=7, linestyle="None", label=name))
            labels.append(name)
    plt.legend(handles, labels)

    plt.title(f"Pareto Fronts for N={N}")
    plt.xlabel("Risk ($f_2$)")
    plt.ylabel("Value ($-f_1$)")
    plt.grid(True, alpha=0.3, zorder=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()



def plot_hv_box(df_metrics, N, out_path):
    """Box plot for HV per method."""
    plt.figure(figsize=(8, 6))
    df_n = df_metrics[df_metrics['N'] == N]
    if df_n.empty:
        plt.title(f'Hypervolume (HV) Distribution for N={N} (No data)')
        plt.savefig(out_path)
        plt.close()
        return

    methods = df_n['method'].unique()
    data = [df_n[df_n['method'] == m]['HV'] for m in methods]

    plt.boxplot(data, labels=methods)
    plt.title(f'Hypervolume (HV) Distribution for N={N}')
    plt.ylabel('HV')
    plt.grid(True, axis='y')
    plt.savefig(out_path)
    plt.close()

def plot_runtime(meta_df, out_path):
    """Bar/box plot over elapsed_s by {N, method}."""
    if meta_df.empty:
        plt.figure(figsize=(10, 6))
        plt.title('Runtime Overview (No data)')
        plt.savefig(out_path)
        plt.close()
        return

    meta_df['N'] = meta_df['N'].astype(str)
    methods = meta_df['method'].unique()
    n_methods = len(methods)
    n_groups = len(meta_df['N'].unique())

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.35
    index = np.arange(n_groups)

    for i, method in enumerate(methods):
        method_data = meta_df[meta_df['method'] == method]
        means = method_data.groupby('N')['elapsed_s'].mean()
        stds = method_data.groupby('N')['elapsed_s'].std().fillna(0)

        # Align data with all N values
        all_ns = sorted(meta_df['N'].unique())
        aligned_means = [means.get(n, 0) for n in all_ns]
        aligned_stds = [stds.get(n, 0) for n in all_ns]

        ax.bar(index + i * bar_width, aligned_means, bar_width, yerr=aligned_stds, capsize=5, label=method)

    # Time cap line
    if 'time_cap_s' in meta_df.columns and not meta_df['time_cap_s'].isnull().all():
        time_cap = meta_df['time_cap_s'].max()
        ax.axhline(y=time_cap, color='r', linestyle='--', label=f'Time Cap ({time_cap}s)')

    ax.set_xlabel('Problem Size (N)')
    ax.set_ylabel('Elapsed Time (s)')
    ax.set_title('Runtime by Problem Size and Method')
    ax.set_xticks(index + bar_width / (2/n_methods) - bar_width/2)
    ax.set_xticklabels(sorted(meta_df['N'].unique()))
    ax.legend()
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
