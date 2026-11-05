import os
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import timedelta

import numpy as np
import pandas as pd
from dateutil.parser import isoparse
from scipy.signal import medfilt

from Functions import functions as f

# Command line arguments
parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("input", type=str, help='The list of dates to analyze in the required .CSV format.')
# The --set_downstream flag takes True or False. True takes the average BSN distance and propagates the data there.
# False propagates it to the BATS-R-US upstream boundary.
parser.add_argument("--set_downstream", default=True, action='store_true',)
parser.add_argument("--bs", choices=['dynamic', 'static'], default="dynamic",
                    help='Choose a dynamically-updated or static bow shock nose. Options are dynamic or static')
args = parser.parse_args()

# Declare important constants. These will be used for manipulating positions/temperatures/etc.
RE = 6371              # Earth radius in kilometers.
l1_dist = 1495980      # L1 distance in km.
kboltz = 1.380649E-23  # Boltzmann constant, J/K
mp = 1.67262192E-27    # Proton mass in Kg
bound_dist = 32 * RE   # Distance to BATS-R-US upstream boundary from Earth.

# Set propagation variable names and units.
swmf_vars = ['bx', 'by', 'bz', 'ux', 'uy', 'uz', 'n', 't']
units = {v: u for v, u in zip(swmf_vars, 3 * ['nT'] + 3 * ['km/s'] + ['cm-3', 'K'])}

# This function reconstructs the upstream solar wind from whichever spacecraft OMNI was using at each point
# The function replaces the old `mfi` and 'swe' loops
# Returns a DataFrame (columns `names`, default `cols`) with an ISO-string Time column,
def reconstruct(omni, ace, wind, cols, names=None):
    t = omni['Time'] - pd.to_timedelta(omni['ts'], unit='s')
    a, w = ace.set_index('Time')[cols], wind.set_index('Time')[cols]

    ace_first = (omni['IMF'] == 71).to_numpy() # Creates a boolean numpy array of the timestamps where ACE is used
    ace_ok, wind_ok = t.isin(a.index).to_numpy(), t.isin(w.index).to_numpy() # Creates two arrays that give T/F values if ACE/Wind data exists. Checks each timestamp in t and returns True if data exists for that time, and False if it is missing.
    use_ace = ace_ok & (ace_first | ~wind_ok)  # This creates a boolean mask where ACE will be used (True) only if ACE data actually exists (ace_ok) AND at least one of the following is true: ACE is flagged as the spacecraft used, OR WIND doesn't exist.
    use_wind = wind_ok & ~use_ace

    vals = np.where(use_ace[:, None], a.reindex(t).to_numpy(),
                    np.where(use_wind[:, None], w.reindex(t).to_numpy(), np.nan))

    names = names or cols
    df = pd.DataFrame(vals, columns=names)
    df.insert(0, 'Time', [x.isoformat() for x in t])
    df = df.sort_values(by='Time')
    df = df.set_index(pd.to_datetime(df['Time']).rename('datetime'))
    df[names] = df[names].interpolate(method='time')  # numeric columns only
    return df.reset_index(drop=True)


# Setting up the event list using arguments
event_list = pd.read_csv(args.input, delimiter=',', header=0)
for e in range(len(event_list)):
    print(f'Run {e + 1} out of {len(event_list)}.')
    start = event_list['Date_Start'][e]
    date_stop = event_list['Date_Stop'][e]
    date_start = pd.to_datetime(start)

    date_string = date_start.strftime('%Y-%m-%d')
    folder_path = f'./Data/Interpolated/{date_string}'

    # Importing in CSV files generated using GetEvents.py
    omni = pd.read_csv(f'{folder_path}/omni.csv', parse_dates=['Time'])
    artemis = pd.read_csv(f'{folder_path}/artemis.csv', parse_dates=['Time'])
    ace_mfi = pd.read_csv(f'{folder_path}/ace_mfi.csv', parse_dates=['Time'])
    ace_swe = pd.read_csv(f'{folder_path}/ace_swe.csv', parse_dates=['Time'])
    wind_mfi = pd.read_csv(f'{folder_path}/wind_mfi.csv', parse_dates=['Time'])
    wind_swe = pd.read_csv(f'{folder_path}/wind_swe.csv', parse_dates=['Time'])

    # Finding the correct shift distance
    if args.set_downstream:
        bound_dist = np.average(omni['BSN'])

    # Reconstruct the OMNI magnetic field and plasma from ACE and Wind
    print('Reconstructing upstream solar wind...')
    mfi = reconstruct(omni, ace_mfi, wind_mfi, ['BX', 'BY', 'BZ'], ['bx', 'by', 'bz'])
    swe = reconstruct(omni, ace_swe, wind_swe, ['VX', 'VY', 'VZ', 'Temp', 'NP'])

    # Begin main propagation script:

    # Create unified time:
    t_swe, t_mag = swe['Time'][:], mfi['Time'][:]
    time = f.unify_time(t_swe, t_mag)
    time = [isoparse(t) for t in time]

    # Convert coordinates:
    vx, vy, vz = f.gse_to_gsm(swe['VX'][:],
                              swe['VY'][:],
                              swe['VZ'][:], swe['Time'][:])

    # Convert temperature
    temp = swe['Temp'][:]

    # Extract plasma parameters
    raw = {'time': time,
           'n': f.pair(t_swe, swe['NP'][:], time, varname='n'),
           't': f.pair(t_swe, temp, time, varname='t'),
           'ux': f.pair(t_swe, vx, time, varname='ux'),
           'uy': f.pair(t_swe, vy, time, varname='uy'),
           'uz': f.pair(t_swe, vz, time, varname='uz')}

    # Extract magnetic field parameters:
    for b in ['bx', 'by', 'bz']:
        raw[b] = f.pair(t_mag, mfi[b][:], raw['time'], varname=b)

    # Create seconds-from-start time array:
    tsec = np.array([(t - raw['time'][0]).total_seconds() for t in raw['time']])

    print('Saving upstream solar wind...')
    # Puts all the raw data into Upstream Dataframe
    upstream = pd.DataFrame({
        'Time': raw['time'],
        'BX': raw['bx'],
        'BY': raw['by'],
        'BZ': raw['bz'],
        'VX': raw['ux'],
        'VY': raw['uy'],
        'VZ': raw['uz'],
        'N': raw['n'],
        'T': raw['t']

    }, columns=['Time', 'BX', 'BY', 'BZ', 'VX', 'VY', 'VZ', 'N', 'T']).replace(to_replace=[-1.00000E+31], value=np.nan)

    upstream_sep = pd.DataFrame({
        'year': [t.strftime('%Y') for t in raw['time']],
        'month': [t.strftime('%m') for t in raw['time']],
        'day': [t.strftime('%d') for t in raw['time']],
        'hour': [t.strftime('%H') for t in raw['time']],
        'min': [t.strftime('%M') for t in raw['time']],
        'sec': [t.strftime('%S') for t in raw['time']],
        'ms': [t.strftime('%f')[:-4] for t in raw['time']],
        'BX': raw['bx'],
        'BY': raw['by'],
        'BZ': raw['bz'],
        'VX': raw['ux'],
        'VY': raw['uy'],
        'VZ': raw['uz'],
        'N': raw['n'],
        'T': raw['t']
    }, columns=['year', 'month', 'day', 'hour', 'min', 'sec', 'ms', 'BX', 'BY', 'BZ', 'VX', 'VY', 'VZ', 'N', 'T']).replace(to_replace=[-1.00000E+31], value=np.nan)

    date_string = date_start.strftime('%Y-%m-%d')
    folder_path = f'./Data/Raw'
    os.makedirs(folder_path, exist_ok=True)
    pd.DataFrame(raw).to_csv(f'{folder_path}/{date_string}.csv', index=False)

    folder_path = f'./Data/Upstream/csv'
    os.makedirs(folder_path, exist_ok=True)
    upstream.to_csv(f'{folder_path}/{date_string}.csv', index=False)

    folder_path = f'./Data/Upstream/dat'
    os.makedirs(folder_path, exist_ok=True)

    with open(f'{folder_path}/{date_string}.dat', "w") as g:
        g.write('#COOR\nGSM\n\n#START\n')
    upstream_sep.to_csv(f'{folder_path}/{date_string}.dat', mode="a", index=False, header=False, sep='\t')
