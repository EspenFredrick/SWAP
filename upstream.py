"""
01_upstream.py  —  SWAP Pipeline Stage 1
=========================================
Fetches ACE/Wind MFI+SWE and OMNI data for each event in the input list,
reconstructs the upstream solar wind at the spacecraft location using OMNI
time-shift metadata, and writes:

  outputs/dat/{date}_upstream.dat   — pre-propagation IMF file (SpacePy)
  data/omni/{date}_omni_data.csv    — raw OMNI data for this event

Usage:
    python 01_upstream.py artemis_near_esline.csv [--set_downstream]
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime
from pathlib import Path
from os import makedirs

import numpy as np
import pandas as pd

from dateutil.parser import isoparse
from spacepy.pybats import ImfInput
from spacepy.datamodel import dmarray

from . import functions as f


# =============================================================================
# Paths  (all relative to the SWAP_tools/ project root)
# =============================================================================

ROOT          = Path(__file__).resolve().parents[3]   # SWAP_tools/
DIR_DATA_OMNI = ROOT / "data"    / "omni"
DIR_DATA_ACE  = ROOT / "data"    / "ace"
DIR_DATA_WIND = ROOT / "data"    / "wind"
DIR_OUT_DAT   = ROOT / "outputs" / "dat"
DIR_SPEDAS    = ROOT / "data"    / "spedas_cache"     # pyspedas download cache

for _d in [DIR_DATA_OMNI, DIR_DATA_ACE, DIR_DATA_WIND, DIR_OUT_DAT, DIR_SPEDAS]:
    _d.mkdir(parents=True, exist_ok=True)

# Redirect pyspedas CDF downloads into data/spedas_cache/ instead of ~/pyspedas_data/
# Must be set before the first pyspedas import that triggers a download.
import os
os.environ["SPEDAS_DATA_DIR"] = str(DIR_SPEDAS)

import pyspedas
from pytplot import get_data


# =============================================================================
# Constants
# =============================================================================

RE         = 6371              # Earth radius [km]
L1_DIST    = 1_495_980         # L1 distance from Earth [km]
K_BOLTZ    = 1.380649e-23      # Boltzmann constant [J/K]
MP         = 1.67262192e-27    # Proton mass [kg]
BOUND_DIST = 32 * RE           # Default BATS-R-US upstream boundary [km]

SWMF_VARS = ["bx", "by", "bz", "ux", "uy", "uz", "n", "t"]
UNITS     = {v: u for v, u in zip(SWMF_VARS, 3*["nT"] + 3*["km/s"] + ["cm-3", "K"])}


# =============================================================================
# Spacecraft lookup helpers
# =============================================================================

def _lookup_mfi(omni_row, ace_mfi, wind_mfi):
    """
    Return (bx, by, bz, sc_id) for one OMNI row.
    Tries the OMNI-flagged spacecraft first, falls back to the other,
    returns NaNs if neither has data at the sought time.
    """
    delta     = pd.Timedelta(omni_row["ts"], unit="s")
    seek_time = omni_row["Time"] - delta
    sc_id     = omni_row["IMF"]

    def _ace():
        m = ace_mfi.loc[ace_mfi["Time"] == seek_time]
        return (m.iloc[0]["BX"], m.iloc[0]["BY"], m.iloc[0]["BZ"], 71) if not m.empty else None

    def _wind():
        m = wind_mfi.loc[wind_mfi["Time"] == seek_time]
        return (m.iloc[0]["BX"], m.iloc[0]["BY"], m.iloc[0]["BZ"], 51) if not m.empty else None

    result = (_ace() or _wind()) if sc_id == 71 else (_wind() or _ace())
    return result if result is not None else (np.nan, np.nan, np.nan, np.nan)


def _lookup_swe(omni_row, ace_swe, wind_swe):
    """
    Return (vx, vy, vz, np, temp, sc_id) for one OMNI row.
    Same fallback logic as _lookup_mfi.
    """
    delta     = pd.Timedelta(omni_row["ts"], unit="s")
    seek_time = omni_row["Time"] - delta
    sc_id     = omni_row["IMF"]

    def _ace():
        m = ace_swe.loc[ace_swe["Time"] == seek_time]
        if m.empty:
            return None
        r = m.iloc[0]
        return r["VX"], r["VY"], r["VZ"], r["NP"], r["Temp"], 71

    def _wind():
        m = wind_swe.loc[wind_swe["Time"] == seek_time]
        if m.empty:
            return None
        r = m.iloc[0]
        return r["VX"], r["VY"], r["VZ"], r["NP"], r["Temp"], 51

    result = (_ace() or _wind()) if sc_id == 71 else (_wind() or _ace())
    return result if result is not None else (np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)


def _resample_to_seconds(df, time_col="Time"):
    """Resample a spacecraft DataFrame to 1-second cadence with interpolation."""
    return (df.set_index(time_col)
              .resample("s").mean()
              .interpolate(method="linear")
              .ffill()
              .reset_index())


# =============================================================================
# Per-event processing function
# =============================================================================

def process_event(start: str, date_stop: str, set_downstream: bool = True):
    """
    Fetch all data, reconstruct upstream solar wind, and write output files
    for a single event window.

    Parameters
    ----------
    start         : ISO 8601 event start string
    date_stop     : ISO 8601 event stop string
    set_downstream: If True, use mean BSN as propagation target
    """
    date_start    = pd.to_datetime(start)
    date_label    = date_start.strftime("%Y-%m-%d")
    pre_buffer    = pd.Timedelta(7200, unit="s")
    shifted_start = (date_start - pre_buffer).strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*54}")
    print(f"  Stage 1 — Upstream:  {start}  →  {date_stop}")
    print(f"{'='*54}")

    # -------------------------------------------------------------------------
    # Fetch OMNI (1-min HRO2)
    # -------------------------------------------------------------------------
    pyspedas.projects.omni.data(
        trange=[start, date_stop], datatype="1min", level="hro2", time_clip=True)

    omni = pd.DataFrame({
        "Time":    get_data("BZ_GSM")[0],
        "BY":      get_data("BY_GSM")[1],
        "BZ":      get_data("BZ_GSM")[1],
        "IMF":     get_data("IMF")[1],
        "ts":      get_data("Timeshift")[1],
        "VX":      get_data("Vx")[1],
        "VY":      get_data("Vy")[1],
        "VZ":      get_data("Vz")[1],
        "Density": get_data("proton_density")[1],
        "Temp":    get_data("T")[1],
        "X":       get_data("x")[1],
        "BSN":     get_data("BSN_x")[1],
    })
    omni["Time"] = pd.to_datetime(omni["Time"], unit="s")
    omni = (omni.set_index("Time")
                .resample("min").mean()
                .interpolate(method="linear")
                .ffill().bfill()
                .reset_index())

    # Save OMNI immediately — it's needed by File 4 independent of propagation
    omni_out = DIR_DATA_OMNI / f"{date_label}_omni_data.csv"
    omni.to_csv(omni_out, index=False)
    print(f"  OMNI saved: {omni_out}")

    # -------------------------------------------------------------------------
    # Fetch ACE MFI + SWE
    # -------------------------------------------------------------------------
    pyspedas.projects.ace.mfi(
        trange=[shifted_start, date_stop], datatype="h3", time_clip=True)
    ace_mfi = pd.DataFrame({
        "Time": get_data("BGSM")[0],
        "BX":   get_data("BGSM")[1][:, 0],
        "BY":   get_data("BGSM")[1][:, 1],
        "BZ":   get_data("BGSM")[1][:, 2],
    }).replace(-1.0e31, np.nan)
    ace_mfi["Time"] = pd.to_datetime(ace_mfi["Time"], unit="s")
    ace_mfi = _resample_to_seconds(ace_mfi)

    pyspedas.projects.ace.swe(
        trange=[shifted_start, date_stop], datatype="h0", time_clip=True)
    ace_swe = pd.DataFrame({
        "Time": get_data("V_GSE")[0],
        "VX":   get_data("V_GSE")[1][:, 0],
        "VY":   get_data("V_GSE")[1][:, 1],
        "VZ":   get_data("V_GSE")[1][:, 2],
        "NP":   get_data("Np")[1],
        "Temp": get_data("Tpr")[1],
    }).replace(-1.0e31, np.nan)
    ace_swe["Time"] = pd.to_datetime(ace_swe["Time"], unit="s")
    ace_swe = _resample_to_seconds(ace_swe)

    # -------------------------------------------------------------------------
    # Fetch Wind MFI + SWE
    # -------------------------------------------------------------------------
    pyspedas.projects.wind.mfi(
        trange=[shifted_start, date_stop], datatype="h0", time_clip=True)
    wind_mfi = pd.DataFrame({
        "Time": get_data("B3GSM")[0],
        "BX":   get_data("B3GSM")[1][:, 0],
        "BY":   get_data("B3GSM")[1][:, 1],
        "BZ":   get_data("B3GSM")[1][:, 2],
    }).replace(-1.0e31, np.nan)
    wind_mfi["Time"] = pd.to_datetime(wind_mfi["Time"], unit="s")
    wind_mfi = _resample_to_seconds(wind_mfi)

    pyspedas.projects.wind.swe(
        trange=[shifted_start, date_stop], datatype="h1",
        varnames=[], time_clip=True)
    wind_temp = (MP / (2 * K_BOLTZ)) * (get_data("Proton_W_moment")[1] * 1000) ** 2
    wind_swe = pd.DataFrame({
        "Time": get_data("Proton_VX_moment")[0],
        "VX":   get_data("Proton_VX_moment")[1],
        "VY":   get_data("Proton_VY_moment")[1],
        "VZ":   get_data("Proton_VZ_moment")[1],
        "NP":   get_data("Proton_Np_moment")[1],
        "Temp": wind_temp,
    }).replace(-1.0e31, np.nan)
    wind_swe["Time"] = pd.to_datetime(wind_swe["Time"], unit="s")
    wind_swe = _resample_to_seconds(wind_swe)

    # -------------------------------------------------------------------------
    # Propagation target distance
    # -------------------------------------------------------------------------
    bound_dist = float(np.mean(omni["BSN"])) if set_downstream else BOUND_DIST

    # -------------------------------------------------------------------------
    # Reconstruct upstream time series
    # -------------------------------------------------------------------------
    mfi_rows = [_lookup_mfi(omni.iloc[i], ace_mfi, wind_mfi) for i in range(len(omni))]
    swe_rows = [_lookup_swe(omni.iloc[i], ace_swe, wind_swe) for i in range(len(omni))]

    up_times = [
        (omni["Time"].iloc[i] - pd.Timedelta(omni["ts"].iloc[i], unit="s")).isoformat()
        for i in range(len(omni))
    ]

    mfi_up = (pd.DataFrame(mfi_rows, columns=["bx", "by", "bz", "sc_id"])
                .assign(Time=up_times)
                .sort_values("Time")
                .assign(datetime=lambda d: pd.to_datetime(d["Time"]))
                .set_index("datetime")
                .interpolate(method="time")
                .reset_index(drop=True))

    swe_up = (pd.DataFrame(swe_rows, columns=["VX", "VY", "VZ", "NP", "Temp", "sc_id"])
                .assign(Time=up_times)
                .sort_values("Time")
                .assign(datetime=lambda d: pd.to_datetime(d["Time"]))
                .set_index("datetime")
                .interpolate(method="time")
                .reset_index(drop=True))

    # -------------------------------------------------------------------------
    # Unify time, convert coordinates, assemble raw dict
    # -------------------------------------------------------------------------
    t_swe = swe_up["Time"][:]
    t_mag = mfi_up["Time"][:]
    time  = [isoparse(t) for t in f.unify_time(t_swe, t_mag)]

    vx_gsm, vy_gsm, vz_gsm = f.gse_to_gsm(
        swe_up["VX"][:], swe_up["VY"][:], swe_up["VZ"][:], t_swe)

    raw = {
        "time": time,
        "n":    f.pair(t_swe, swe_up["NP"][:],    time, varname="n"),
        "t":    f.pair(t_swe, swe_up["Temp"][:],  time, varname="t"),
        "ux":   f.pair(t_swe, vx_gsm,             time, varname="ux"),
        "uy":   f.pair(t_swe, vy_gsm,             time, varname="uy"),
        "uz":   f.pair(t_swe, vz_gsm,             time, varname="uz"),
        "bx":   f.pair(t_mag, mfi_up["bx"][:],    time, varname="bx"),
        "by":   f.pair(t_mag, mfi_up["by"][:],    time, varname="by"),
        "bz":   f.pair(t_mag, mfi_up["bz"][:],    time, varname="bz"),
    }

    if "PGSM" in mfi_up.columns:
        raw["pos"] = f.pair(t_mag, mfi_up["PGSM"][:, 0], time, varname="pos") * RE

    if "pos" in raw:
        print("  S/C location found — using dynamic location.")
        raw["X"] = raw["pos"] - bound_dist
    else:
        print("  S/C location NOT found — using static L1 distance.")
        raw["X"] = L1_DIST - bound_dist

    # -------------------------------------------------------------------------
    # Write upstream IMF file
    # -------------------------------------------------------------------------
    out_path = str(DIR_OUT_DAT / f"{date_label}_upstream.dat")
    imf_up   = ImfInput(out_path, load=False, npoints=len(time))

    for v in SWMF_VARS:
        imf_up[v] = dmarray(raw[v], {"units": UNITS[v]})
    imf_up["time"] = np.array(raw["time"])
    imf_up.attrs["header"].append("Reconstructed upstream solar wind at spacecraft location\n")
    imf_up.attrs["header"].append("Source: ACE (id=71) / Wind (id=51/52) via OMNI timeshift\n")
    imf_up.attrs["header"].append("\n")
    imf_up.attrs["coor"]   = "GSM"
    imf_up.attrs["satxyz"] = [np.mean(raw["X"]) / RE, 0, 0]
    imf_up.attrs["header"].append(f"File created on {datetime.now()}\n")
    imf_up.write()
    print(f"  Upstream IMF written: {out_path}")

    return raw, omni, date_label


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("input", type=str,
                        help="Event list CSV with Date_Start and Date_Stop columns.")
    parser.add_argument("--set_downstream", default=True, action="store_true",
                        help="Use mean BSN as propagation target (default: True).")
    args = parser.parse_args()

    event_list = pd.read_csv(args.input, delimiter=",", header=0)
    for e in range(len(event_list) - 1):
        process_event(
            event_list["Date_Start"][e],
            event_list["Date_Stop"][e],
            set_downstream=args.set_downstream,
        )
