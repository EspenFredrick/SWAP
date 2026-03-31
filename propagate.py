"""
02_propagate.py  —  SWAP Pipeline Stage 2
==========================================
Reads the upstream IMF file written by 01_upstream.py, then runs both:
  (A) Ballistic propagation to the BATS-R-US upstream boundary
  (B) 1-D ideal MHD propagation from L1 to the bow shock

Writes:
  outputs/dat/{date}_downstream.dat     — ballistically propagated IMF
  outputs/csv/{date}_mhd_input.csv      — MHD-solver-compatible CSV
  outputs/csv/{date}_mhd_output.csv     — MHD solver results
  outputs/plots/{date}_propagation.png  — 5-panel comparison: ballistic, MHD, OMNI

Usage:
    python 02_propagate.py artemis_near_esline.csv [--target_re 12.0]
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import medfilt
from scipy.interpolate import interp1d
from matplotlib.lines import Line2D
from spacepy.pybats import ImfInput
from spacepy.datamodel import dmarray
from spacepy.plot import applySmartTimeTicks


# =============================================================================
# Paths
# =============================================================================

ROOT        = Path(__file__).resolve().parents[3]
DIR_OUT_DAT  = ROOT / "outputs" / "dat"
DIR_OUT_CSV  = ROOT / "outputs" / "csv"
DIR_OUT_PLOT = ROOT / "outputs" / "plots"
DIR_DATA_OMNI = ROOT / "data"   / "omni"

for _d in [DIR_OUT_DAT, DIR_OUT_CSV, DIR_OUT_PLOT]:
    _d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Constants
# =============================================================================

RE         = 6371
L1_DIST    = 1_495_980
K_BOLTZ    = 1.380649e-23
MP         = 1.67262192e-27
BOUND_DIST = 32 * RE

SWMF_VARS = ["bx", "by", "bz", "ux", "uy", "uz", "n", "t"]
UNITS     = {v: u for v, u in zip(SWMF_VARS, 3*["nT"] + 3*["km/s"] + ["cm-3", "K"])}

# MHD solver physical constants (SI)
MU_0  = 4.0 * np.pi * 1e-7
M_P   = 1.6726e-27
K_B   = 1.3806e-23
R_E   = 6.371e6
GAMMA = 5.0 / 3.0
EPS   = 1.0e-30


# =============================================================================
# Part A — Ballistic propagation
# =============================================================================

def ballistic_propagate(upstream_dat_path: str, bound_dist: float, date_label: str):
    """
    Load an upstream IMF .dat file, apply ballistic time-shifting to propagate
    to the BATS-R-US upstream boundary, and write a downstream .dat file.

    Returns the ImfInput object for the downstream file and the raw dict
    (for plotting).
    """
    imf_up = ImfInput(upstream_dat_path, load=True)

    raw = {v: np.array(imf_up[v]) for v in SWMF_VARS}
    raw["time"] = np.array(imf_up["time"])

    # Spacecraft distance from propagation target
    sc_x_km = imf_up.attrs.get("satxyz", [L1_DIST / RE])[0] * RE
    X        = sc_x_km - bound_dist

    print(f"  Ballistic: propagation distance = {X:.0f} km")
    vel_smooth = medfilt(raw["ux"], 1)
    shift_sec  = X / vel_smooth

    t_shifted = np.array([
        t1 - timedelta(seconds=float(t2))
        for t1, t2 in zip(raw["time"], shift_sec)
    ])

    # Remove overtaken points
    keep   = [0]
    last_t = t_shifted[0]
    for i in range(1, len(raw["time"])):
        if t_shifted[i] > last_t:
            keep.append(i)
            last_t = t_shifted[i]
    print(f"  Removed {len(raw['time']) - len(keep)} overtaken points.")

    out_path   = str(DIR_OUT_DAT / f"{date_label}_downstream.dat")
    imf_down   = ImfInput(out_path, load=False, npoints=len(keep))
    for v in SWMF_VARS:
        imf_down[v] = dmarray(raw[v][keep], {"units": UNITS[v]})
    imf_down["time"] = t_shifted[keep]
    imf_down.attrs["header"].append(
        "Ballistically propagated from L1 to upstream BATS-R-US boundary\n")
    imf_down.attrs["coor"]   = "GSM"
    imf_down.attrs["satxyz"] = imf_up.attrs.get("satxyz", [0, 0, 0])
    imf_down.attrs["header"].append(f"File created on {datetime.now()}\n")
    imf_down.write()
    print(f"  Downstream IMF written: {out_path}")

    return imf_down, raw


# =============================================================================
# Part B — 1-D MHD solver
# =============================================================================

def _dat_to_mhd_csv(upstream_dat_path: str, csv_out_path: str) -> pd.DataFrame:
    """
    Convert a SpacePy ImfInput .dat file to the CSV format the MHD solver expects.
    Column mapping: bx->Bx, by->By, bz->Bz, ux->Vx, uy->Vy, uz->Vz, n->n, t->T
    Time is converted to Unix epoch float seconds.
    """
    imf = ImfInput(upstream_dat_path, load=True)

    raw_time = imf["time"]
    try:
        t_epoch = raw_time.astype("datetime64[s]").astype(np.float64)
    except (AttributeError, TypeError):
        t_epoch = np.array([t.timestamp() for t in raw_time], dtype=np.float64)

    # Enforce monotonic time (required by interp1d)
    keep_mask = np.concatenate([[True], np.diff(t_epoch) > 0])
    if not keep_mask.all():
        print(f"  MHD CSV: dropping {(~keep_mask).sum()} non-monotonic time steps.")
    t_epoch = t_epoch[keep_mask]

    df = pd.DataFrame({
        "time": t_epoch,
        "Bx":   np.array(imf["bx"])[keep_mask],
        "By":   np.array(imf["by"])[keep_mask],
        "Bz":   np.array(imf["bz"])[keep_mask],
        "Vx":   np.array(imf["ux"])[keep_mask],
        "Vy":   np.array(imf["uy"])[keep_mask],
        "Vz":   np.array(imf["uz"])[keep_mask],
        "n":    np.array(imf["n"])[keep_mask],
        "T":    np.array(imf["t"])[keep_mask],
    })

    vx_vals = df["Vx"].dropna()
    if (vx_vals > 0).mean() > 0.5:
        print("  Warning: >50% of Vx values are positive — check GSM sign convention.")

    df.to_csv(csv_out_path, index=False)
    print(f"  MHD input CSV written: {csv_out_path}  ({len(df)} rows)")
    return df


# --- MHD numerics ---

def _prim_to_cons(rho, vx, vy, vz, by, bz, p, bx_c):
    kin  = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    mag  = (bx_c**2 + by**2 + bz**2) / (2.0 * MU_0)
    return np.array([rho, rho*vx, rho*vy, rho*vz, by, bz,
                     p / (GAMMA - 1.0) + kin + mag])


def _cons_to_prim(U, bx_c):
    rho  = max(U[0], EPS)
    vx, vy, vz = U[1]/rho, U[2]/rho, U[3]/rho
    by, bz     = U[4], U[5]
    kin  = 0.5 * rho * (vx**2 + vy**2 + vz**2)
    mag  = (bx_c**2 + by**2 + bz**2) / (2.0 * MU_0)
    p    = max((GAMMA - 1.0) * (U[6] - kin - mag), EPS)
    return rho, vx, vy, vz, by, bz, p


def _mhd_flux(U, bx_c):
    rho, vx, vy, vz, by, bz, p = _cons_to_prim(U, bx_c)
    B2    = bx_c**2 + by**2 + bz**2
    p_tot = p + B2 / (2.0 * MU_0)
    BdotV = bx_c*vx + by*vy + bz*vz
    F     = np.empty(7)
    F[0]  = rho * vx
    F[1]  = rho*vx*vx + p_tot - bx_c**2 / MU_0
    F[2]  = rho*vx*vy - bx_c*by / MU_0
    F[3]  = rho*vx*vz - bx_c*bz / MU_0
    F[4]  = by*vx - bx_c*vy
    F[5]  = bz*vx - bx_c*vz
    F[6]  = (U[6] + p_tot)*vx - bx_c*BdotV / MU_0
    return F


def _hll_flux(U_L, U_R, bx_c):
    def _speeds(U):
        rho, vx, _, _, by, bz, p = _cons_to_prim(U, bx_c)
        B2  = bx_c**2 + by**2 + bz**2
        v_A = np.sqrt(B2 / (MU_0 * max(rho, EPS)))
        c_s = np.sqrt(GAMMA * max(p, EPS) / max(rho, EPS))
        c_f = np.sqrt(v_A**2 + c_s**2)
        return vx - c_f, vx + c_f

    sLm, sLp = _speeds(U_L)
    sRm, sRp = _speeds(U_R)
    s_L, s_R = min(sLm, sRm), max(sLp, sRp)

    if s_L >= 0:
        return _mhd_flux(U_L, bx_c)
    if s_R <= 0:
        return _mhd_flux(U_R, bx_c)

    F_L = _mhd_flux(U_L, bx_c)
    F_R = _mhd_flux(U_R, bx_c)
    return (s_R*F_L - s_L*F_R + s_L*s_R*(U_R - U_L)) / (s_R - s_L)


def _compute_dt(U_grid, bx_c, dx, cfl=0.5):
    max_speed = EPS
    for U in U_grid:
        rho, vx, _, _, by, bz, p = _cons_to_prim(U, bx_c)
        B2  = bx_c**2 + by**2 + bz**2
        v_A = np.sqrt(B2 / (MU_0 * max(rho, EPS)))
        c_s = np.sqrt(GAMMA * max(p, EPS) / max(rho, EPS))
        max_speed = max(max_speed, abs(vx) + np.sqrt(v_A**2 + c_s**2))
    return cfl * dx / max_speed


def run_mhd_solver(csv_path: str, target_re: float = 12.0, nx: int = 300,
                   cfl: float = 0.5, output_interval: float = 60.0):
    """
    Solve 1-D ideal MHD from L1 to the bow shock.

    Returns
    -------
    results : ndarray (n_output, 8) — [time, rho, vx, vy, vz, By, Bz, p]
    t_data  : 1-D float array of input times [s]
    interps : dict of scipy interp1d objects keyed by field name
    """
    df = pd.read_csv(csv_path)
    t  = df["time"].values.astype(float)

    fields = {
        "bx":  df["Bx"].values * 1e-9,
        "by":  df["By"].values * 1e-9,
        "bz":  df["Bz"].values * 1e-9,
        "vx":  df["Vx"].values * 1e3,
        "vy":  df["Vy"].values * 1e3,
        "vz":  df["Vz"].values * 1e3,
        "rho": df["n"].values * 1e6 * M_P,
        "p":   df["n"].values * 1e6 * K_B * df["T"].values,
    }
    interps = {k: interp1d(t, v, kind="linear", fill_value="extrapolate")
               for k, v in fields.items()}

    # Grid
    L1_m   = 1.5e9
    BS_m   = target_re * R_E
    x_span = L1_m - BS_m
    dx     = x_span / nx

    # Initialise
    bx0 = float(interps["bx"](t[0]))
    U   = np.zeros((nx, 7))
    U[:] = _prim_to_cons(float(interps["rho"](t[0])), float(interps["vx"](t[0])),
                         float(interps["vy"](t[0])),  float(interps["vz"](t[0])),
                         float(interps["by"](t[0])),  float(interps["bz"](t[0])),
                         float(interps["p"](t[0])),   bx0)
    bx_c = bx0

    t_now, next_out = t[0], t[0]
    results, steps  = [], 0

    print("  Running MHD solver...")
    while t_now < t[-1]:
        # Boundary conditions
        bx_c = float(interps["bx"](t_now))
        U[0] = _prim_to_cons(float(interps["rho"](t_now)), float(interps["vx"](t_now)),
                              float(interps["vy"](t_now)),  float(interps["vz"](t_now)),
                              float(interps["by"](t_now)),  float(interps["bz"](t_now)),
                              float(interps["p"](t_now)),   bx_c)
        U[-1] = U[-2].copy()

        dt = min(_compute_dt(U, bx_c, dx, cfl), t[-1] - t_now,
                 next_out - t_now + 1e-6)

        fluxes = np.array([_hll_flux(U[i], U[i+1], bx_c) for i in range(nx-1)])
        U_new  = U.copy()
        U_new[1:-1] = U[1:-1] - (dt/dx) * (fluxes[1:] - fluxes[:-1])

        for i in range(1, nx-1):
            U_new[i, 0] = max(U_new[i, 0], EPS)
            r, vx, vy, vz, by, bz, p = _cons_to_prim(U_new[i], bx_c)
            U_new[i] = _prim_to_cons(r, vx, vy, vz, by, bz, p, bx_c)

        U      = U_new
        t_now += dt
        steps += 1

        if t_now >= next_out:
            r, vx, vy, vz, by, bz, p = _cons_to_prim(U[-1], bx_c)
            results.append([t_now, r, vx, vy, vz, by, bz, p])
            next_out += output_interval

        if steps % 10000 == 0:
            pct = 100.0 * (t_now - t[0]) / (t[-1] - t[0])
            print(f"    t = {t_now:.0f} s  ({pct:.1f}%)")

    print(f"  MHD done — {steps} steps, {len(results)} output points.")
    return np.array(results), t, interps


# =============================================================================
# Part C — Combined plot (ballistic + MHD + OMNI)
# =============================================================================

def plot_combined(raw_up, imf_down, mhd_results, mhd_interps, omni_df,
                  date_label: str):
    """
    5-panel figure:
      Row 1: By   — upstream (dashed), ballistic (solid), MHD (dotted), OMNI (grey)
      Row 2: Bz
      Row 3: Vx
      Row 4: n (number density)
      Row 5: T (temperature)
    """
    mhd_t   = mhd_results[:, 0]       # Unix epoch seconds
    mhd_rho = mhd_results[:, 1]
    mhd_vx  = mhd_results[:, 2]
    mhd_by  = mhd_results[:, 5]
    mhd_bz  = mhd_results[:, 6]
    mhd_p   = mhd_results[:, 7]
    mhd_n   = mhd_rho / M_P / 1e6
    mhd_T   = mhd_p / (mhd_n * 1e6 * K_B)

    t_down  = imf_down["time"]   # datetime objects
    t_up    = raw_up["time"]

    omni_t  = omni_df["Time"]

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=False)
    fig.suptitle(f"Solar Wind Propagation Comparison  —  {date_label}", fontsize=13)

    panels = [
        ("by", "By  [nT]",    raw_up["by"],  imf_down["by"],
         mhd_by * 1e9,        omni_df["BY"]),
        ("bz", "Bz  [nT]",    raw_up["bz"],  imf_down["bz"],
         mhd_bz * 1e9,        omni_df["BZ"]),
        ("ux", "Vx  [km/s]",  raw_up["ux"],  imf_down["ux"],
         mhd_vx / 1e3,        omni_df["VX"]),
        ("n",  "n  [cm⁻³]",   raw_up["n"],   imf_down["n"],
         mhd_n,               omni_df["Density"]),
        ("t",  "T  [K]",      raw_up["t"],   imf_down["t"],
         mhd_T,               omni_df["Temp"]),
    ]

    for ax, (_, ylabel, y_up, y_down, y_mhd, y_omni) in zip(axes, panels):
        ax.plot(omni_t,  y_omni,  color="0.7",      lw=1.0, label="OMNI")
        ax.plot(t_up,    y_up,    color="steelblue", lw=1.0, ls="--", alpha=0.7,
                label="Upstream (reconstructed)")
        ax.plot(t_down,  y_down,  color="steelblue", lw=1.5, label="Ballistic")
        ax.plot(mhd_t,   y_mhd,   color="tomato",    lw=1.5, label="1-D MHD")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    legend_handles = [
        Line2D([], [], color="0.7",      lw=1.5, label="OMNI"),
        Line2D([], [], color="steelblue", lw=1.0, ls="--", label="Upstream"),
        Line2D([], [], color="steelblue", lw=1.5, label="Ballistic"),
        Line2D([], [], color="tomato",    lw=1.5, label="1-D MHD"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4, fontsize=9)
    fig.subplots_adjust(hspace=0.06, top=0.93, bottom=0.05, right=0.97)

    out_path = DIR_OUT_PLOT / f"{date_label}_propagation.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# =============================================================================
# Per-event runner
# =============================================================================

def process_event(date_label: str, target_re: float = 12.0,
                  set_downstream: bool = True):
    """
    Run both propagation methods for one event date and produce combined plot.
    Upstream .dat and OMNI .csv must already exist (written by 01_upstream.py).
    """
    upstream_path = str(DIR_OUT_DAT   / f"{date_label}_upstream.dat")
    omni_path     = str(DIR_DATA_OMNI / f"{date_label}_omni_data.csv")
    csv_path      = str(DIR_OUT_CSV   / f"{date_label}_mhd_input.csv")
    mhd_out_path  = str(DIR_OUT_CSV   / f"{date_label}_mhd_output.csv")

    omni_df = pd.read_csv(omni_path, parse_dates=["Time"])

    bound_dist = float(omni_df["BSN"].mean()) if set_downstream else BOUND_DIST

    print(f"\n{'='*54}")
    print(f"  Stage 2 — Propagate:  {date_label}")
    print(f"{'='*54}")

    # A — Ballistic
    imf_down, raw_up = ballistic_propagate(upstream_path, bound_dist, date_label)

    # B — MHD
    _dat_to_mhd_csv(upstream_path, csv_path)
    mhd_results, t_data, mhd_interps = run_mhd_solver(
        csv_path, target_re=target_re)

    header = "time_s,rho_kg_m3,vx_m_s,vy_m_s,vz_m_s,By_T,Bz_T,p_Pa"
    np.savetxt(mhd_out_path, mhd_results, delimiter=",", header=header)
    print(f"  MHD output written: {mhd_out_path}")

    # C — Plot
    plot_combined(raw_up, imf_down, mhd_results, mhd_interps, omni_df, date_label)


# =============================================================================
# CLI entry point
# =============================================================================

if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__,
                            formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("input", type=str,
                        help="Event list CSV (same file used for 01_upstream.py).")
    parser.add_argument("--target_re", type=float, default=12.0,
                        help="Bow shock standoff distance in Earth radii (default 12).")
    parser.add_argument("--set_downstream", default=True, action="store_true")
    args = parser.parse_args()

    event_list = pd.read_csv(args.input, delimiter=",", header=0)
    for e in range(len(event_list) - 1):
        date_label = pd.to_datetime(event_list["Date_Start"][e]).strftime("%Y-%m-%d")
        process_event(date_label, target_re=args.target_re,
                      set_downstream=args.set_downstream)
