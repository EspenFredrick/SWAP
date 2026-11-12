import warnings
warnings.filterwarnings('ignore')

import os
import traceback
from argparse import ArgumentParser, RawDescriptionHelpFormatter
from datetime import timedelta

import numpy as np
import pandas as pd
from dateutil.parser import isoparse

from Functions import functions as f
from Functions import GetSatellitesGSM as GS

from multiprocessing import Pool

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

# Function that prints what is going on in the terminal, so you know that the program isn't hanging
# It just prints the message you put in (msg) followed by a new line and 52 tildes (~)
def say(msg):
    print(f'{msg}\n{"~" * 52}')

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

INPUT_FILES = ['omni', 'artemis', 'ace_mfi', 'ace_swe', 'wind_mfi', 'wind_swe']

# Returns the yyyy-mm-dd string for an event, used to label errors
def event_date(e):
    return pd.to_datetime(event_list['Date_Start'][e]).strftime('%Y-%m-%d')

# Processes one event. Returns (status, date, message), where status is 'ok' or 'error'
def process_event(e):
    print(f'Running event {e + 1} of {len(event_list)}', flush=True)
    start = event_list['Date_Start'][e]
    date_stop = event_list['Date_Stop'][e]
    date_start = pd.to_datetime(start)

    date_string = date_start.strftime('%Y-%m-%d')
    folder_path = f'./Data/Interpolated/{date_string}'

    # Check that every input file from the download script exists
    missing = [n for n in INPUT_FILES if not os.path.exists(f'{folder_path}/{n}.csv')]
    if missing:
        return ('error', date_string, f'missing input files: {", ".join(missing)}')

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

    keys = ['bx', 'by', 'bz', 'ux', 'uy', 'uz', 'n', 't']
    secs = (pd.to_datetime(raw['time']) - pd.Timestamp(0)).total_seconds()  # Convert datetimes to Unix seconds, which regrid expects
    df = GS.regrid(secs, {k: np.asarray(raw[k], dtype=float) for k in keys}, 'min', bad=-1e30)  # Resample onto a regular 1-minute grid, interpolate gaps, and mask leftover fill values

    # Note (but do not stop for) NaNs left in the regridded data
    notes = []
    nan_by_key = {k: np.isnan(df[k].to_numpy(dtype=float)).mean() for k in keys}
    nan_by_key = {k: v for k, v in nan_by_key.items() if v > 0}
    if nan_by_key:
        notes.append('NaNs after regrid: ' + ', '.join(f'{k} {v:.1%}' for k, v in nan_by_key.items()))

    raw = {'time': list(df.index.to_pydatetime()), **{k: df[k].to_numpy() for k in keys}}   # Rebuild raw with the regular 1-minute times and the regridded values

    # Pad end of input so the last real data reaches the satellite
    dt = timedelta(minutes=1)   # Use the minute cadence
    t_end = pd.to_datetime(omni['Time'].iloc[-1]).to_pydatetime()
    n_pad = int((t_end - raw['time'][-1]) / dt) # Find the number of pad points needed to match OMNI
    new_t = [raw['time'][-1] + dt * (i + 1) for i in range(n_pad)]
    for k in keys:
        hold = np.nanmedian(np.asarray(raw[k][-10:], dtype=float))   # Creates the pad values using the median of the last 10 values
        raw[k] = list(raw[k]) + [hold] * n_pad  # Repeat the held value for every padded point
    raw['time'] = list(raw['time']) + new_t # Append the padded timestamps

    # Create seconds-from-start time array:
    tsec = np.array([(t - raw['time'][0]).total_seconds() for t in raw['time']])

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

    uptime = pd.to_datetime(upstream['Time'])   # Make a copy of the time column as a list of datetime objs
    date_parts = (uptime.dt.strftime('%Y %m %d %H %M %S %f').str.split(' ', expand=True))   # Create a string with the parts and then split the string based on whitespace
    date_parts.columns = ['year', 'month', 'day', 'hour', 'min', 'sec', 'ms']
    date_parts['ms'] = date_parts['ms'].str[:-4]  # Truncate the milliseconds
    upstream_sep = pd.concat([date_parts, upstream.drop(columns='Time')], axis=1)   # Copy "upstream", drop the time col, and add the parts

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

    print(f'Finished processing event {e + 1}, padded {n_pad} values (of {len(upstream)} total)', flush=True)
    return ('ok', date_string, '; '.join(notes))

# Wraps process_event so one bad event doesn't kill the whole batch
def safe_process(e):
    try:
        return process_event(e)
    except Exception as ex:
        tb = traceback.extract_tb(ex.__traceback__)[-1]
        where = f'{os.path.basename(tb.filename)}:{tb.lineno} in {tb.name}'
        print(f'FAILED event {e + 1}', flush=True)
        return ('error', event_date(e), f'{type(ex).__name__}: {ex} [{where}]')

# Prints (and saves) the summary block at the end of the run
# 'ok' results with a message are files that were written but had something worth noting
def summarize(results, name):
    ok = [r for r in results if r[0] == 'ok']
    warn = sorted((r for r in ok if r[2]), key=lambda r: r[1])
    err = sorted((r for r in results if r[0] == 'error'), key=lambda r: r[1])
    bar = '~' * 52
    lines = [bar,
             f'Wrote {len(ok)} files of {len(results)} in the event list.',
             f'{len(err)} errors, {len(ok)} successful ({len(warn)} with warnings)']
    if err:
        lines.append('errors (no file written):')
        lines += [f'{d}: {m}' for _, d, m in err]
    if warn:
        lines.append('warnings (file written anyway):')
        lines += [f'{d}: {m}' for _, d, m in warn]
    lines.append(bar)
    text = '\n'.join(lines)
    print('\n' + text)
    with open(f'{name}_log.txt', 'w') as fh:
        fh.write(text + '\n')

if __name__ == '__main__':
    with Pool(processes=min(32, len(event_list)), maxtasksperchild=1) as pool:
        results = pool.map(safe_process, range(len(event_list)), chunksize=1)
    summarize(results, 'CalculateUpstream')