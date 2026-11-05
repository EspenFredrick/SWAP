# Import Statements
import numpy as np
import pandas as pd
import pyspedas
from pytplot import get_data
from scipy.constants import physical_constants
import os

KB_EV = physical_constants['Boltzmann constant in eV/K'][0]
RM = 1737.4                         # lunar radius [km]
KEYS = ['BX_GSM', 'BY_GSM', 'BZ_GSM', 'VX', 'VY', 'VZ', 'N', 'T']
LEAD = pd.Timedelta(minutes=60)     # ARTEMIS sees the wind before OMNI's BSN timestamp
FMT = '%Y-%m-%d/%H:%M:%S'

def _get(name):
    d = get_data(name)
    return d[0], d[1]

# Set {name: array} to a regualar cadence (grid), interpolate gaps, fill the front/back values.
# Values <= `bad` (e.g. -1e30 fill values) are treated as missing.
def regrid(time, cols, freq, bad=None): # `bad` is usually replaced by whatever the NaN fill is (usually FILL)
    df = pd.DataFrame(cols, index=pd.to_datetime(time, unit='s').rename('Time'))    # Create `Time` column
    if bad is not None:
        df = df.mask(df <= bad) # `mask` replaces values in a DataFrame where a given condition is True
    return df.resample(freq).mean().interpolate().ffill().bfill()   # Interpolate and fill step

# B field, ion moments and position for one probe, on a common 1-min grid.
def _probe(p, trange, stop):
    pyspedas.themis.fgm(trange=trange, probe=p, level='l2', time_clip=True,
                        varnames=f'th{p}_fgs_gsm')
    t, b = _get(f'th{p}_fgs_gsm')
    fgm = regrid(t, {'BX_GSM': b[:, 0], 'BY_GSM': b[:, 1], 'BZ_GSM': b[:, 2]}, 'min')

    names = [f'th{p}_peif_velocity_gsm', f'th{p}_peif_density', f'th{p}_peif_avgtemp']
    pyspedas.themis.esa(trange=trange, probe=p, level='l2', time_clip=True, varnames=names)
    t, v = _get(names[0])
    esa = regrid(t, {'VX': v[:, 0], 'VY': v[:, 1], 'VZ': v[:, 2],
                     'N': _get(names[1])[1],
                     'T': _get(names[2])[1] / KB_EV}, 'min')  # eV -> K

    names = [f'th{p}_pos_gsm', f'th{p}_pos_sse']
    pyspedas.themis.state(trange=trange, probe=p, time_clip=True, varnames=names)
    t, gsm = _get(names[0])
    sse = _get(names[1])[1]
    stt = regrid(t, {'XPOS': gsm[:, 0], 'XSSE': sse[:, 0] / RM, 'YSSE': sse[:, 1] / RM}, 'min')

    frames = [fgm, esa, stt]
    t0 = max(f.index[0] for f in frames)  # datasets can start at different times
    return pd.concat(frames, axis=1).loc[t0:stop].ffill().bfill()

# ARTEMIS solar wind between start and stop (datetimes), indexed by Time.
# Data start LEAD before `start`. Where probe B is in the lunar wake (XSSE <= 0 and |YSSE| <= 1 R_moon), its values are replaced by probe C's."
def get_artemis_gsm(start, stop):
    trange = [(start - LEAD).strftime(FMT), (stop + pd.Timedelta(minutes=1)).strftime(FMT)]
    b, c = (_probe(p, trange, stop) for p in 'bc')

    wake = (b['XSSE'] <= 0) & b['YSSE'].between(-1, 1) & b.index.isin(c.index)
    b.loc[wake, KEYS] = c.loc[b.index[wake], KEYS].values
    return b