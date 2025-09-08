
import os
os.environ["MPLBACKEND"] = "Agg"   # headless matplotlib to avoid Qt warnings

import pandas as pd
from datetime import datetime

from metrics import (
    load_fronts, load_meta, hv_igd_tables, aggregate_ci
)
from plots import (
    plot_pareto, plot_hv_box, plot_runtime
)

FRONTS_DIR = "multiobj_outputs/fronts"
LOGS_DIR = "multiobj_outputs/logs"
OUTPUT_DIR = "report/out"

def main():
    """Main script to generate the report."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df_fronts = load_fronts(FRONTS_DIR)
    meta_df = load_meta(LOGS_DIR)

    if df_fronts.empty or meta_df.empty:
        report_content = "# Experiment Report\n\n**Warning: No data found in `multiobj_outputs`. Report generation skipped.**"
        with open(os.path.join(OUTPUT_DIR, "REPORT.md"), "w") as f:
            f.write(report_content)
        print("No data found. Generated a placeholder report.")
        return

    # Compute metrics
    metrics_per_run = hv_igd_tables(df_fronts)
    metrics_agg = aggregate_ci(metrics_per_run)

    # Save metric CSVs
    metrics_per_run.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics_per_run.csv"), index=False)
    metrics_agg.to_csv(os.path.join(OUTPUT_DIR, "summary_metrics_agg.csv"), index=False)

    # Generate plots
    unique_Ns = sorted(df_fronts['N'].unique())
    for n in unique_Ns:
        plot_pareto(df_fronts, n, os.path.join(OUTPUT_DIR, f"pareto_N{n}.png"))
        plot_hv_box(metrics_per_run, n, os.path.join(OUTPUT_DIR, f"hv_box_N{n}.png"))

    plot_runtime(meta_df, os.path.join(OUTPUT_DIR, "runtime.png"))

    # Generate Markdown Report
    generate_markdown_report(meta_df, metrics_agg, unique_Ns)


def generate_markdown_report(meta_df, metrics_agg, unique_Ns):
    """Generates the final Markdown report file."""
    # Lists of Ns and methods actually present in the metrics table
    Ns = sorted(metrics_agg['N'].unique().tolist())
    methods = sorted(metrics_agg['method'].unique().tolist())

    def render_metric_table(metric: str, is_int: bool = False) -> str:
        # Build a Markdown table as pure strings (no numeric DF to avoid dtype issues)
        header = "| N | " + " | ".join(methods) + " |\n"
        sep = "|" + "|".join(["---"] * (len(methods) + 1)) + "|\n"
        lines = [header, sep]

        for N in Ns:
            row_cells = []
            for m in methods:
                row = metrics_agg[(metrics_agg["N"] == N) & (metrics_agg["method"] == m)]
                if row.empty:
                    cell = "—"
                else:
                    mean = row[f"{metric}_mean"].iloc[0]
                    lo   = row[f"{metric}_ci_low"].iloc[0]
                    hi   = row[f"{metric}_ci_high"].iloc[0]
                    if pd.isna(mean) or pd.isna(lo) or pd.isna(hi):
                        cell = "—"
                    else:
                        if is_int:
                            cell = f"{mean:.0f} ({lo:.0f} - {hi:.0f})"
                        else:
                            cell = f"{mean:.3f} ({lo:.3f} - {hi:.3f})"
                row_cells.append(cell)
            lines.append("| " + str(N) + " | " + " | ".join(row_cells) + " |\n")

        return "".join(lines)

    report_content = f"""# Multi-Objective Experiment Report

*Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

## 1. Setup

This report summarizes the performance of multi-objective optimization methods.

- **Problem Sizes (N):** {sorted(meta_df['N'].unique())}
- **Methods:** {sorted(meta_df['method'].unique())}
- **Seeds:** {sorted(meta_df['seed'].unique())}
- **Total Runs:** {len(meta_df)}
- **Time Caps (s):** {sorted(meta_df['time_cap_s'].dropna().unique().tolist())}

**Objective Interpretation:**
- **Value:** `-f1` (higher is better)
- **Risk:** `f2` (lower is better)

The goal is to find solutions that maximize Value while minimizing Risk, representing a classic Pareto trade-off.

## 2. Performance Metrics

Metrics are aggregated across seeds (mean ± 95% CI).

- **HV (Hypervolume) ↑:** Measures the volume of the dominated portion of the objective space. Higher is better.
- **IGD+ (Inverted Generational Distance Plus) ↓:** Measures the average distance from each point in the reference front to the obtained front. Lower is better.
- **|ND| (Number of Non-Dominated Points) ↑:** The number of points in the final Pareto front. Higher is generally better, indicating more choices.

### HV (↑) mean ± 95% CI

"""

    report_content += render_metric_table("HV", is_int=False)
    report_content += "\n### IGD+ (↓) mean ± 95% CI\n\n"
    report_content += render_metric_table("IGD", is_int=False)
    report_content += "\n### |ND| (↑) mean ± 95% CI\n\n"
    report_content += render_metric_table("ND_size", is_int=True)

    report_content += """\n## 3. Pareto Fronts

Scatter plots of **Risk vs. Value**.  
The ideal region is the top-left (low risk, high value).  
NSGA-II is expected to produce fronts that dominate the random search, demonstrating its effectiveness.

"""

    for n in unique_Ns:
        report_content += f"### N = {n}\n![Pareto Front for N={n}](pareto_N{n}.png)\n"

    report_content += """\n## 4. Runtime Overview

The following plot shows the mean elapsed time per run, with error bars representing the standard deviation across seeds.

![Runtime Overview](runtime.png)

"""

    runtime_summary = meta_df.groupby(['N', 'method'])['elapsed_s'].agg(['mean', 'std']).reset_index()
    report_content += "### Mean Runtime (s) ± Std Dev\n\n" \
                      + runtime_summary.to_markdown(index=False, floatfmt=".2f") + "\n"

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "REPORT.md"), "w") as f:
        f.write(report_content)



if __name__ == "__main__":
    main()
    print(f"Report generated in {OUTPUT_DIR}/")
