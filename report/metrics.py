
import pandas as pd
import numpy as np
import glob
import json
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from pymoo.indicators.igd_plus import IGDPlus

def load_fronts(fronts_dir):
    """Reads all front CSVs and concatenates them."""
    files = glob.glob(f"{fronts_dir}/*.csv")
    if not files:
        return pd.DataFrame()
    df_list = [pd.read_csv(f) for f in files]
    return pd.concat(df_list, ignore_index=True)

def load_meta(logs_dir):
    """Reads all JSON meta."""
    files = glob.glob(f"{logs_dir}/*.json")
    if not files:
        return pd.DataFrame()
    meta_list = []
    for f in files:
        with open(f, 'r') as file:
            meta_list.append(json.load(file))
    return pd.DataFrame(meta_list)

def non_dominated(F):
    """Returns the non-dominated set from F."""
    return F[NonDominatedSorting().do(F, only_non_dominated_front=True)]

def pooled_reference_front(df_fronts, N):
    """Pools all methods & seeds at size N and returns the non-dominated set."""
    df_n = df_fronts[df_fronts['N'] == N]
    if df_n.empty:
        return np.array([])
    pooled_front = df_n[['f1', 'f2']].values
    return non_dominated(pooled_front)

def hv_igd_tables(df_fronts):
    """Computes HV, IGD+, and |ND| for each run."""
    results = []
    if df_fronts.empty:
        return pd.DataFrame(columns=['N', 'method', 'seed', 'HV', 'IGD', 'ND_size'])

    for n_val in df_fronts['N'].unique():
        ref_front = pooled_reference_front(df_fronts, n_val)
        if ref_front.shape[0] == 0:
            continue

        max_vals = np.max(ref_front, axis=0)
        ref_point = max_vals * 1.1
        
        igd_metric = IGDPlus(ref_front)
        hv_metric = HV(ref_point=ref_point)

        for (method, seed), group in df_fronts[df_fronts['N'] == n_val].groupby(['method', 'seed']):
            front = group[['f1', 'f2']].values
            nd_front = non_dominated(front)
            
            if nd_front.shape[0] > 0:
                hv = hv_metric.do(nd_front)
                igd = igd_metric.do(nd_front)
                nd_size = len(nd_front)
            else:
                hv = 0
                igd = np.inf
                nd_size = 0

            results.append({
                'N': n_val,
                'method': method,
                'seed': seed,
                'HV': hv,
                'IGD': igd,
                'ND_size': nd_size
            })
            
    return pd.DataFrame(results)

def aggregate_ci(df, by=["N", "method"]):
    """Computes mean and 95% CI."""
    
    def ci(series):
        n = len(series)
        if n == 0:
            return 0
        mean = series.mean()
        std = series.std()
        ci_val = 1.96 * std / np.sqrt(n)
        return ci_val

    agg_funcs = {
        'HV': ['mean', ci],
        'IGD': ['mean', ci],
        'ND_size': ['mean', ci]
    }
    
    summary = df.groupby(by).agg(agg_funcs).reset_index()
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    
    for metric in ['HV', 'IGD', 'ND_size']:
        summary[f'{metric}_ci_low'] = summary[f'{metric}_mean'] - summary[f'{metric}_ci']
        summary[f'{metric}_ci_high'] = summary[f'{metric}_mean'] + summary[f'{metric}_ci']
        summary = summary.drop(columns=[f'{metric}_ci'])

    return summary.rename(columns={'N_': 'N', 'method_': 'method'})
