"""
04_correlate.py  —  SWAP Pipeline Stage 4
==========================================
Correlates propagated solar wind time series (upstream, ballistic, 1-D MHD)
against ARTEMIS in-situ observations using sliding 60-minute windows.

For each event, for each variable, for each method:
  - Slides a 60-minute Artemis window against a fixed propagated window
    over 31 lag steps (0–30 minutes)
  - Records peak Pearson r, prediction efficiency (PE), optimal lag, and
    expected ballistic arrival time
  - Saves per-day metrics and lag-shift CSVs, then merges all days into a
    single summary file per method/variable

Output structure under outputs/correlations/:
    {method}/
        {variable}/
            metrics/    — one CSV per event day
            shifts/     — one CSV per event day (optimal lag per window)
            merged/     — output.csv concatenating all days

Usage:
    python 04_correlate.py artemis_near_esline.csv
    python 04_correlate.py artemis_near_esline.csv --variables BZ_GSM BY_GSM VX
    python 04_correlate.py artemis_near_esline.csv --no_pe --parallel
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# =============================================================================
# Paths
# =============================================================================

ROOT          = Path(__file__).resolve().parents[3]
DIR_DATA_ART  = ROOT / "data"    / "artemis"
DIR_DATA_OMNI = ROOT / "data"    / "omni"
DIR_OUT_CSV   = ROOT / "outputs" / "csv"
DIR_OUT_CORR  = ROOT / "outputs" / "correlations"

for _d in [DIR_OUT_CORR, DIR_OUT_CSV]:
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Metric functions  (from doCorrelate.py)
# =============================================================================

def prediction_efficiency_artemis(omni: np.ndarray, artemis: np.ndarray) -> float:
    """PE using Artemis mean as baseline — measures how well the propagated
    signal predicts in-situ observations."""
    return 1 - (np.sum((omni - artemis) ** 2) /
                np.sum((artemis - np.mean(artemis)) ** 2))


def prediction_efficiency_omni(omni: np.ndarray, artemis: np.ndarray) -> float:
    """PE using propagated signal mean as baseline."""
    return 1 - (np.sum((artemis - omni) ** 2) /
                np.sum((omni - np.mean(omni)) ** 2))


def find_max_correlation(values: list) -> tuple:
    """
    Return (peak_value, index) from a list of correlation values.
    Skips index 0 (zero-lag). If all non-zero-lag values are negative,
    returns the one with the largest absolute value.
    """
    tail = values[1:]
    if all(c < 0 for c in tail):
        best_val = max(tail, key=abs)
    else:
        best_val = max(tail)
    return best_val, values.index(best_val)


# =============================================================================
# Core sliding-window correlation
# =============================================================================

def correlate(file_pair: tuple, output_path: str, variable: str = "BZ_GSM",
              include_pe: bool = True) -> None:
    """
    Sliding 60-minute window correlation between an Artemis DataFrame and a
    propagated (OMNI/ballistic/MHD) DataFrame.

    Both DataFrames must already be resampled to 1-minute cadence on a
    shared time axis (done by _process_pair before calling this).

    Required columns: 'Time', the target variable, 'XPOS', 'VX'.

    Parameters
    ----------
    file_pair   : (artemis_df, pred_df) — matched 1-min DataFrames.
    output_path : root output directory; sub-folders are created automatically.
    variable    : column name to correlate (e.g. 'BZ_GSM', 'VX', 'V').
    include_pe  : if True, also compute prediction efficiency metrics.
    """
    artemis, omni = file_pair
    num_windows   = len(omni) - 60

    print(f"    Computing {num_windows} windows for {variable}...")

    data_rows   = []
    offset_rows = []

    for n in range(num_windows):
        # Match Artemis index to the current OMNI window start/stop times
        try:
            art_start = artemis.loc[artemis["Time"] == omni["Time"].iloc[n]].index[0]
            art_stop  = artemis.loc[artemis["Time"] == omni["Time"].iloc[n + 59]].index[0]
        except IndexError:
            continue   # Gap in Artemis coverage — skip this window

        # Positional metadata for ballistic comparison
        o_xpos        = np.average(omni["XPOS"].iloc[n:n + 59])
        a_xpos        = np.average(artemis["XPOS"].iloc[art_start:art_stop])
        hourly_offset = a_xpos - o_xpos
        hourly_vel    = np.average(omni["VX"].iloc[n:n + 59])
        with np.errstate(divide="ignore", invalid="ignore"):
            pred_arrival = int((hourly_offset / np.abs(hourly_vel)) / 60) \
                           if hourly_vel != 0 else 0

        # Build all 31 lag-shifted Artemis slices
        o_slice  = omni[variable].iloc[n:n + 59].values
        a_shifts = [artemis[variable].iloc[art_start - i:art_stop - i].values
                    for i in range(31)]

        # Vectorised correlations across all lags
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            pearson_vals = [
                float(np.corrcoef(o_slice, a)[0, 1])
                if len(a) == len(o_slice) else np.nan
                for a in a_shifts
            ]

        # Replace any NaNs with 0 so find_max_correlation works cleanly
        pearson_clean = [v if np.isfinite(v) else 0.0 for v in pearson_vals]
        peak_r, peak_idx = find_max_correlation(pearson_clean)

        if include_pe:
            pe_a = prediction_efficiency_artemis(o_slice, a_shifts[peak_idx])
            pe_o = prediction_efficiency_omni(o_slice, a_shifts[peak_idx])
            data_rows.append([
                omni["Time"].iloc[n], omni["Time"].iloc[n + 59],
                peak_r, pe_a, pe_o,
                hourly_offset, hourly_vel, pred_arrival,
            ])
            offset_rows.append([
                omni["Time"].iloc[n], omni["Time"].iloc[n + 60],
                peak_r, pe_a, pe_o, peak_idx, pred_arrival,
            ])
        else:
            data_rows.append([
                omni["Time"].iloc[n], omni["Time"].iloc[n + 59],
                peak_r, hourly_offset, hourly_vel, pred_arrival,
            ])
            offset_rows.append([
                omni["Time"].iloc[n], omni["Time"].iloc[n + 60],
                peak_r, peak_idx, pred_arrival,
            ])

    if include_pe:
        values = pd.DataFrame(data_rows, columns=[
            "Start", "Stop", "Pearson", "PE_Artemis", "PE_Omni",
            "hourly-position", "hourly-velocity", "expected-arrival",
        ])
        shifts = pd.DataFrame(offset_rows, columns=[
            "Start", "Stop", "Pearson", "PE_Artemis", "PE_Omni",
            "lag-minutes", "expected-arrival",
        ])
    else:
        values = pd.DataFrame(data_rows, columns=[
            "Start", "Stop", "Pearson",
            "hourly-position", "hourly-velocity", "expected-arrival",
        ])
        shifts = pd.DataFrame(offset_rows, columns=[
            "Start", "Stop", "Pearson", "lag-minutes", "expected-arrival",
        ])

    metrics_dir = os.path.join(output_path, variable, "metrics")
    shifts_dir  = os.path.join(output_path, variable, "shifts")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(shifts_dir,  exist_ok=True)

    date_str = artemis["Time"].iloc[0].strftime("%Y-%m-%d")
    values.to_csv(os.path.join(metrics_dir, f"{date_str}.csv"), index=False)
    shifts.to_csv(os.path.join(shifts_dir,  f"{date_str}.csv"), index=False)


# =============================================================================
# Data loading helpers
# =============================================================================

def _load_artemis_csv(date_label: str, probe: str) -> pd.DataFrame | None:
    """Load and merge ARTEMIS FGM + ESA + position CSVs for one probe."""
    fgm_p = DIR_DATA_ART / f"{date_label}_artemis_{probe}_fgm.csv"
    esa_p = DIR_DATA_ART / f"{date_label}_artemis_{probe}_esa.csv"
    pos_p = DIR_DATA_ART / f"{date_label}_artemis_{probe}_pos.csv"

    if not fgm_p.exists():
        return None

    fgm = pd.read_csv(fgm_p, parse_dates=["Time"])
    fgm = fgm.rename(columns={"BX": "BX_GSM", "BY": "BY_GSM", "BZ": "BZ_GSM"})

    if esa_p.exists():
        esa = pd.read_csv(esa_p, parse_dates=["Time"])
        esa["Time"] = esa["Time"].dt.round("s")
        fgm["Time"] = fgm["Time"].dt.round("s")
        esa["V"]    = np.sqrt(esa["VX"]**2 + esa["VY"]**2 + esa["VZ"]**2)
        fgm = fgm.merge(esa[["Time", "VX", "VY", "VZ", "NP", "Temp", "V"]],
                        on="Time", how="outer")
        fgm = fgm.rename(columns={"NP": "N", "Temp": "T"})

    if pos_p.exists():
        pos = pd.read_csv(pos_p, parse_dates=["Time"])
        pos["Time"] = pos["Time"].dt.round("min")
        fgm["Time"] = fgm["Time"].dt.round("s")
        fgm = fgm.merge(pos[["Time", "X_GSM"]], on="Time", how="left")
        fgm = fgm.rename(columns={"X_GSM": "XPOS"})
    else:
        fgm["XPOS"] = np.nan

    return fgm.sort_values("Time").reset_index(drop=True)


def _load_propagated(date_label: str, method: str) -> pd.DataFrame | None:
    """
    Load a propagated time series as a standardised DataFrame.
    method: 'omni' | 'ballistic' | 'mhd'

    Standardised output columns:
        Time, BX_GSM, BY_GSM, BZ_GSM, VX, VY, VZ, N, T, V, XPOS
    """
    if method == "omni":
        p = DIR_DATA_OMNI / f"{date_label}_omni_data.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, parse_dates=["Time"])
        df = df.rename(columns={"Density": "N", "Temp": "T",
                                 "BY": "BY_GSM", "BZ": "BZ_GSM"})
        df["BX_GSM"] = np.nan
        df["V"]      = np.sqrt(df["VX"]**2 + df["VY"]**2 + df["VZ"]**2)
        # Use BSN (bow shock nose x-position) as the OMNI spatial reference
        df["XPOS"]   = df.get("BSN", df.get("X", np.nan))

    elif method == "ballistic":
        from spacepy.pybats import ImfInput
        p = ROOT / "outputs" / "dat" / f"{date_label}_downstream.dat"
        if not p.exists():
            return None
        imf = ImfInput(str(p), load=True)
        try:
            t = pd.to_datetime(imf["time"].astype("datetime64[s]"))
        except Exception:
            t = pd.to_datetime(imf["time"])
        df = pd.DataFrame({
            "Time":   t,
            "BX_GSM": np.array(imf["bx"]),
            "BY_GSM": np.array(imf["by"]),
            "BZ_GSM": np.array(imf["bz"]),
            "VX":     np.array(imf["ux"]),
            "VY":     np.array(imf["uy"]),
            "VZ":     np.array(imf["uz"]),
            "N":      np.array(imf["n"]),
            "T":      np.array(imf["t"]),
        })
        df["V"]    = np.sqrt(df["VX"]**2 + df["VY"]**2 + df["VZ"]**2)
        df["XPOS"] = imf.attrs.get("satxyz", [0])[0] * 6371.0  # Re -> km

    elif method == "mhd":
        p = DIR_OUT_CSV / f"{date_label}_mhd_output.csv"
        if not p.exists():
            return None
        df = pd.read_csv(p, comment="#",
                         names=["time_s", "rho", "vx", "vy", "vz",
                                "By_T", "Bz_T", "p_Pa"])
        MP_  = 1.6726e-27
        KB_  = 1.3806e-23
        n    = df["rho"] / MP_ / 1e6
        df   = pd.DataFrame({
            "Time":   pd.to_datetime(df["time_s"], unit="s"),
            "BX_GSM": np.nan,
            "BY_GSM": df["By_T"] * 1e9,
            "BZ_GSM": df["Bz_T"] * 1e9,
            "VX":     df["vx"]   / 1e3,
            "VY":     df["vy"]   / 1e3,
            "VZ":     df["vz"]   / 1e3,
            "N":      n,
            "T":      df["p_Pa"] / (n * 1e6 * KB_),
        })
        df["V"]    = np.sqrt(df["VX"]**2 + df["VY"]**2 + df["VZ"]**2)
        df["XPOS"] = np.nan   # MHD output is at bow shock, no L1 position

    else:
        raise ValueError(f"Unknown method '{method}'. Use omni, ballistic, or mhd.")

    return df.sort_values("Time").reset_index(drop=True)


# =============================================================================
# Task runner  (called directly or via ProcessPoolExecutor)
# =============================================================================

def _process_pair(args: tuple) -> None:
    """
    Prepare one (date, probe, method, variable) correlation task and run it.
    Resamples both DataFrames to 1-minute cadence on their shared time overlap
    before passing to correlate().
    """
    date_label, probe, method, variable, corr_dir, include_pe = args

    art_df  = _load_artemis_csv(date_label, probe)
    pred_df = _load_propagated(date_label, method)

    if art_df is None or pred_df is None:
        print(f"  Skip {date_label} probe={probe} method={method}: missing file.")
        return

    if variable not in art_df.columns or variable not in pred_df.columns:
        print(f"  Skip {date_label} probe={probe} method={method}: "
              f"'{variable}' not in both DataFrames.")
        return

    # Resample to 1-minute cadence on shared time overlap
    t_start = max(art_df["Time"].min(), pred_df["Time"].min())
    t_end   = min(art_df["Time"].max(), pred_df["Time"].max())
    if t_start >= t_end:
        print(f"  Skip {date_label} probe={probe} method={method}: no time overlap.")
        return

    common_t = pd.date_range(t_start, t_end, freq="1min")

    def _resamp(df: pd.DataFrame) -> pd.DataFrame:
        numeric = df.select_dtypes(include="number").columns.tolist()
        return (df.set_index("Time")[numeric]
                  .reindex(common_t)
                  .interpolate(method="time", limit_direction="both")
                  .reset_index()
                  .rename(columns={"index": "Time"}))

    art_r  = _resamp(art_df)
    pred_r = _resamp(pred_df)

    # Ensure V column exists
    for df in [art_r, pred_r]:
        if "V" not in df.columns and all(c in df.columns for c in ["VX","VY","VZ"]):
            df["V"] = np.sqrt(df["VX"]**2 + df["VY"]**2 + df["VZ"]**2)

    # Ensure XPOS exists (needed by correlate() for positional metadata)
    for df in [art_r, pred_r]:
        if "XPOS" not in df.columns:
            df["XPOS"] = 0.0

    method_dir = os.path.join(corr_dir, method)
    os.makedirs(method_dir, exist_ok=True)

    print(f"\n  {date_label}  probe={probe}  method={method}  var={variable}")
    correlate((art_r, pred_r), method_dir, variable=variable,
              include_pe=include_pe)


# =============================================================================
# Merge helper
# =============================================================================

def merge_daily_csvs(method_dir: str, variable: str,
                     subfolder: str = "metrics") -> None:
    """
    Concatenate all per-day CSVs in {method_dir}/{variable}/{subfolder}/
    into a single merged output.csv.
    """
    src = os.path.join(method_dir, variable, subfolder)
    if not os.path.isdir(src):
        return

    files = sorted(f for f in os.listdir(src)
                   if not f.startswith(".") and f.endswith(".csv"))
    if not files:
        return

    merged = pd.concat(
        [pd.read_csv(os.path.join(src, f), header=0) for f in files],
        ignore_index=True)

    out_dir = os.path.join(method_dir, variable, "merged")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "output.csv")
    merged.to_csv(out_path, index=False)
    print(f"  Merged {len(files)} day-files → {out_path}  ({len(merged)} rows)")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("input", type=str,
                        help="Event list CSV (Date_Start / Date_Stop columns).")
    parser.add_argument("--variables", nargs="+",
                        default=["BZ_GSM", "BY_GSM", "BX_GSM", "VX", "N", "T"],
                        help="Variables to correlate.")
    parser.add_argument("--methods", nargs="+",
                        default=["omni", "ballistic", "mhd"],
                        help="Propagation methods to compare.")
    parser.add_argument("--probes", nargs="+", default=["b", "c"],
                        help="ARTEMIS probes (b = THEMIS-B, c = THEMIS-C).")
    parser.add_argument("--no_pe", action="store_true",
                        help="Skip prediction efficiency (faster).")
    parser.add_argument("--parallel", action="store_true",
                        help="Parallelise using ProcessPoolExecutor.")
    args = parser.parse_args()

    event_list = pd.read_csv(args.input, delimiter=",", header=0)
    corr_dir   = str(DIR_OUT_CORR)
    include_pe = not args.no_pe

    # Build full task list
    tasks = [
        (pd.to_datetime(event_list["Date_Start"][e]).strftime("%Y-%m-%d"),
         probe, method, variable, corr_dir, include_pe)
        for e in range(len(event_list) - 1)
        for probe    in args.probes
        for method   in args.methods
        for variable in args.variables
    ]

    print(f"\nTotal tasks : {len(tasks)}")
    print(f"Variables   : {args.variables}")
    print(f"Methods     : {args.methods}")
    print(f"Probes      : {args.probes}")
    print(f"PE          : {include_pe}")
    print(f"Parallel    : {args.parallel}\n")

    if args.parallel:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_process_pair, t) for t in tasks]
            for fut in futures:
                try:
                    fut.result()
                except Exception as exc:
                    print(f"  Task failed: {exc}")
    else:
        for t in tasks:
            _process_pair(t)

    # Merge all per-day CSVs into summary files
    print("\nMerging daily output files...")
    for method in args.methods:
        method_dir = os.path.join(corr_dir, method)
        for variable in args.variables:
            merge_daily_csvs(method_dir, variable, "metrics")
            merge_daily_csvs(method_dir, variable, "shifts")

    print("\nStage 4 complete.")
