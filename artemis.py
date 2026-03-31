"""
03_artemis.py  —  SWAP Pipeline Stage 3
=========================================
Fetches ARTEMIS (THEMIS-B and THEMIS-C) magnetic field and plasma data
for each event in the input list and saves them to data/artemis/.

ARTEMIS probes are THEMIS-B (probe='b') and THEMIS-C (probe='c').
This script fetches FGM (fluxgate magnetometer) and ESA (electrostatic
analyser) data, converts to GSM coordinates, and writes one CSV per probe
per event.

Output files (data/artemis/):
    {date}_artemis_b_fgm.csv   — THEMIS-B magnetic field (GSM, nT)
    {date}_artemis_b_esa.csv   — THEMIS-B plasma moments
    {date}_artemis_c_fgm.csv   — THEMIS-C magnetic field (GSM, nT)
    {date}_artemis_c_esa.csv   — THEMIS-C plasma moments

Usage:
    python 03_artemis.py artemis_near_esline.csv
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import os
import numpy as np
import pandas as pd


# =============================================================================
# Paths
# =============================================================================

ROOT         = Path(__file__).resolve().parents[3]
DIR_DATA_ART = ROOT / "data" / "artemis"
DIR_SPEDAS   = ROOT / "data" / "spedas_cache"

for _d in [DIR_DATA_ART, DIR_SPEDAS]:
    _d.mkdir(parents=True, exist_ok=True)

# Redirect pyspedas CDF downloads — must be set before the first download call
os.environ["SPEDAS_DATA_DIR"] = str(DIR_SPEDAS)

import pyspedas
from pytplot import get_data


# =============================================================================
# Constants
# =============================================================================

K_BOLTZ = 1.380649e-23
MP      = 1.67262192e-27
PROBES  = ["b", "c"]   # THEMIS-B and THEMIS-C are the ARTEMIS probes


# =============================================================================
# Fetch helpers
# =============================================================================

def _fetch_fgm(probe: str, trange: list) -> pd.DataFrame:
    """
    Fetch THEMIS FGM (fluxgate magnetometer) data in GSM coordinates
    for one probe and return as a tidy DataFrame.

    Columns: Time, BX, BY, BZ  [nT, GSM]
    """
    varname = f"th{probe}_fgs_gsm"
    pyspedas.projects.themis.fgm(
        probe=probe, trange=trange, datatype="fgs",
        coord="gsm", time_clip=True)

    data = get_data(varname)
    if data is None:
        print(f"  FGM: no data for probe {probe.upper()} in {trange}")
        return pd.DataFrame(columns=["Time", "BX", "BY", "BZ"])

    df = pd.DataFrame({
        "Time": pd.to_datetime(data[0], unit="s"),
        "BX":   data[1][:, 0],
        "BY":   data[1][:, 1],
        "BZ":   data[1][:, 2],
    }).replace(-1.0e31, np.nan)

    df = (df.set_index("Time")
            .resample("s").mean()
            .interpolate(method="linear")
            .ffill()
            .reset_index())

    return df


def _fetch_esa(probe: str, trange: list) -> pd.DataFrame:
    """
    Fetch THEMIS ESA (electrostatic analyser) ion moment data for one probe.
    Computes temperature from thermal velocity the same way as Wind SWE.

    Columns: Time, VX, VY, VZ  [km/s, GSM], NP [cm⁻³], Temp [K]
    """
    pyspedas.projects.themis.esa(
        probe=probe, trange=trange, datatype="peif",
        time_clip=True)

    # Velocity in GSM
    vel_var  = f"th{probe}_peif_velocity_gsm"
    dens_var = f"th{probe}_peif_density"
    vth_var  = f"th{probe}_peif_vthermal"

    vel  = get_data(vel_var)
    dens = get_data(dens_var)
    vth  = get_data(vth_var)

    if vel is None or dens is None:
        print(f"  ESA: no data for probe {probe.upper()} in {trange}")
        return pd.DataFrame(columns=["Time", "VX", "VY", "VZ", "NP", "Temp"])

    # Temperature from thermal speed: T = mp * vth^2 / (2 * kb)
    # vth is in km/s — convert to m/s first
    if vth is not None:
        temp = (MP / (2.0 * K_BOLTZ)) * (vth[1] * 1e3) ** 2
        t_temp = vth[0]
    else:
        temp   = np.full(len(vel[0]), np.nan)
        t_temp = vel[0]

    df_vel = pd.DataFrame({
        "Time": pd.to_datetime(vel[0],  unit="s"),
        "VX":   vel[1][:, 0],
        "VY":   vel[1][:, 1],
        "VZ":   vel[1][:, 2],
    })
    df_dens = pd.DataFrame({
        "Time": pd.to_datetime(dens[0], unit="s"),
        "NP":   dens[1],
    })
    df_temp = pd.DataFrame({
        "Time": pd.to_datetime(t_temp,  unit="s"),
        "Temp": temp,
    })

    # Merge on nearest-second cadence
    for df in [df_vel, df_dens, df_temp]:
        df["Time"] = df["Time"].dt.round("s")

    df = (df_vel.merge(df_dens, on="Time", how="outer")
                .merge(df_temp, on="Time", how="outer")
                .replace(-1.0e31, np.nan)
                .sort_values("Time"))

    df = (df.set_index("Time")
            .resample("s").mean()
            .interpolate(method="linear")
            .ffill()
            .reset_index())

    return df


def _fetch_pos(probe: str, trange: list) -> pd.DataFrame:
    """
    Fetch THEMIS spacecraft position in GSM [Re].

    Columns: Time, X_GSM, Y_GSM, Z_GSM  [Re]
    """
    pyspedas.projects.themis.state(
        probe=probe, trange=trange, time_clip=True)

    pos_var = f"th{probe}_pos_gsm"
    pos     = get_data(pos_var)

    if pos is None:
        print(f"  STATE: no position data for probe {probe.upper()}")
        return pd.DataFrame(columns=["Time", "X_GSM", "Y_GSM", "Z_GSM"])

    RE_KM = 6371.0
    df = pd.DataFrame({
        "Time":  pd.to_datetime(pos[0], unit="s"),
        "X_GSM": pos[1][:, 0] / RE_KM,
        "Y_GSM": pos[1][:, 1] / RE_KM,
        "Z_GSM": pos[1][:, 2] / RE_KM,
    }).replace(-1.0e31, np.nan)

    df = (df.set_index("Time")
            .resample("min").mean()
            .interpolate(method="linear")
            .ffill()
            .reset_index())

    return df


# =============================================================================
# Per-event processing
# =============================================================================

def process_event(start: str, date_stop: str):
    """
    Fetch ARTEMIS data for both probes over one event window and write CSVs.
    """
    date_label = pd.to_datetime(start).strftime("%Y-%m-%d")
    trange     = [start, date_stop]

    print(f"\n{'='*54}")
    print(f"  Stage 3 — ARTEMIS:  {start}  →  {date_stop}")
    print(f"{'='*54}")

    for probe in PROBES:
        print(f"\n  Probe TH-{probe.upper()}:")

        # FGM
        fgm = _fetch_fgm(probe, trange)
        if not fgm.empty:
            out = DIR_DATA_ART / f"{date_label}_artemis_{probe}_fgm.csv"
            fgm.to_csv(out, index=False)
            print(f"    FGM saved: {out}  ({len(fgm)} rows)")

        # ESA
        esa = _fetch_esa(probe, trange)
        if not esa.empty:
            out = DIR_DATA_ART / f"{date_label}_artemis_{probe}_esa.csv"
            esa.to_csv(out, index=False)
            print(f"    ESA saved: {out}  ({len(esa)} rows)")

        # Position
        pos = _fetch_pos(probe, trange)
        if not pos.empty:
            out = DIR_DATA_ART / f"{date_label}_artemis_{probe}_pos.csv"
            pos.to_csv(out, index=False)
            print(f"    Position saved: {out}  ({len(pos)} rows)")


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("input", type=str,
                        help="Event list CSV with Date_Start and Date_Stop columns.")
    args = parser.parse_args()

    event_list = pd.read_csv(args.input, delimiter=",", header=0)
    for e in range(len(event_list) - 1):
        process_event(event_list["Date_Start"][e], event_list["Date_Stop"][e])
