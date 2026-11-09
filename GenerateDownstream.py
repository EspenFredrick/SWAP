import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=FutureWarning)

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
from Functions.GetSatellitesGSM import FMT
import numpy as np
import pyspedas
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from datetime import datetime, timedelta
from spacepy.pybats import ImfInput
from matplotlib.lines import Line2D
from spacepy.plot import applySmartTimeTicks
from spacepy.datamodel import dmarray
from pytplot import get_data
import os

import matplotlib
matplotlib.use('Agg')
from multiprocessing import Pool

from Functions import GetSatellitesGSM as GS

parser = ArgumentParser(description=__doc__,
                        formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("input", type=str, help='The list of dates to analyze in the required .CSV format.')
#parser.add_argument("output", type=str, help='The data output file name.')
# The --set_downstream flag takes True or False. True takes the average BSN distance and propagates the data there.
# False propagates it to the BATS-R-US upstream boundary.
parser.add_argument("--set_downstream", default=True, action='store_true',)
parser.add_argument("--bs", choices=['dynamic', 'static'], default="dynamic",
                    help='Choose a dynamically-updated or static bow shock nose. Options are dynamic or static')
args = parser.parse_args()


# Declare important constants
RE = 6371              # Earth radius in kmeters.
l1_dist = 1495980      # L1 distance in km.
kboltz = 1.380649E-23  # Boltzmann constant, J/K
mp = 1.67262192E-27    # Proton mass in Kg
bound_dist = 32*RE   # Distance to BATS-R-US upstream boundary from Earth.

# Set propagation variable names and units.
swmf_vars = ['bx', 'by', 'bz', 'ux', 'uy', 'uz', 'n', 't']
units = {v: u for v, u in zip(swmf_vars, 3*['nT']+3*['km/s']+['cm-3', 'K'])}

def get(name):
    d = get_data(name)
    return d[0], d[1]

# Function that prints what is going on in the terminal, so you know that the program isn't hanging
# It just prints the message you put in (msg) followed by a new line and 52 tildes (~)
def say(msg):
    print(f'{msg}\n{"~" * 52}')


# Import the OMNI, ACE, and Wind data using the list of times
event_list = pd.read_csv(args.input, delimiter=',', header=0)

def process_event(e):
    print(f'Running event {e + 1} of {len(event_list)}', flush=True)
    start = event_list['Date_Start'][e]
    date_stop = event_list['Date_Stop'][e]
    date_stop = pd.to_datetime(date_stop)
    date_start = pd.to_datetime(start)

    date_string = date_start.strftime('%Y-%m-%d')
    folder_path = f'./Data/Raw'
    date_string = date_start.strftime('%Y-%m-%d')
    raw = pd.read_csv(f'{folder_path}/{date_string}.csv',parse_dates=['time'])

    # Get S/C distance. If not in file, use approximation.
    #print(f'Run {e+1} out of {len(event_list)}.')
    pyspedas.projects.omni.data(trange=[(date_start - timedelta(minutes=180)).strftime(FMT), date_stop.strftime(FMT)],
                                    datatype='1min', level='hro2', time_clip=True)
    t = get('BZ_GSM')[0]
    omni = GS.regrid(t, {'Time':pd.to_datetime(get('BZ_GSM')[0],unit='s'),'BY_GSM': get('BY_GSM')[1], 'BZ_GSM': get('BZ_GSM')[1],
                             'IMF': get('IMF')[1], 'ts': get('Timeshift')[1],
                             'VX': get('Vx')[1], 'VY': get('Vy')[1], 'VZ': get('Vz')[1],
                             'N': get('proton_density')[1], 'T': get('T')[1],
                             'XPOS': get('x')[1], 'BSN': get('BSN_x')[1]}, 'min')

    n = len(raw)    # Length of the raw timeseries
    if args.bs == 'dynamic':
        bound_dist = np.full(n, np.nanmean(omni['BSN']) * RE)  # Create an initial BSN position guess for each point, the average OMNI BSN position over the interval
    else:
        bound_dist = np.full(n, 32 * RE) # Create an array full of 32RE distances (SWMF upstream boundary)

    max_iterations = 10
    tolerance = 0.5 * RE  # Stop looping if BSN shifts by less than 0.5 Earth Radii

    sc_x = raw['pos'].to_numpy() if 'pos' in raw else np.full(n, l1_dist) # Make a numpy array of spacecraft positions if the position is given, otherwise make an array assuming L1 distances
    ux = np.abs(raw['ux'].to_numpy())  # Create numpy array of velocity in km/s; abs() so GSE-negative Vx doesn't flip the sign
    t0 = pd.to_datetime(raw['time']).to_numpy() # Create numpy array of timestamps for each point
    omni_t = pd.to_datetime(omni['Time']).to_numpy() # Create numpy array of OMNI timestamps
    omni_bsn = omni['BSN'].to_numpy() * RE  # Create numpy array of OMNI BSN locations, in km

    active = np.ones(n, dtype=bool)  # Boolean array containing T/F, where True = not yet converged

    if args.bs == 'dynamic':
        print(f"Starting dynamic propagation for event {e+1}...")
        for iteration in range(max_iterations):
            idx = np.flatnonzero(active)
            if idx.size == 0:
                print(f"Convergence achieved for event {e+1}!")
                break

            # distance to guessed BSN, travel time, arrival time (active points only)
            X = sc_x[idx] - bound_dist[idx]
            tshift = X / ux[idx]
            arrival = t0[idx] + (tshift * 1e9).astype('timedelta64[ns]')
            arrival = pd.DatetimeIndex(arrival).ceil('min').to_numpy()

            # OMNI BSN at arrival time
            loc = np.clip(np.searchsorted(omni_t, arrival), 0, len(omni_t) - 1)
            actual = omni_bsn[loc]
            actual = np.where(np.isnan(actual), bound_dist[idx], actual)  # missing OMNI -> keep guess

            # compare guess vs actual; update only the ones that missed
            diff = np.abs(actual - bound_dist[idx])
            bound_dist[idx] = actual
            active[idx[diff < tolerance]] = False

            #print(f"Iteration {iteration + 1}: {active.sum()} points left, "
                  #f"max change {diff.max() / RE:.3f} RE")
        else:
            print(f"Warning for event {e+1}: {active.sum()} points did not converge in {max_iterations} iterations.")
            #say(f"Largest error in convergence was {np.max(np.abs(diff))/RE:.3f} RE")

    raw['X'] = sc_x - bound_dist
    raw['tshift'] = raw['X'] / ux

    # Velocity smoothing (window must be odd; 1 = no smoothing)
    win = 1
    #print(f'Applying smoothing using a {win} window size.')
    velsmooth = medfilt(np.abs(raw['ux'].to_numpy()), win)  # km/s, no RE factor

    #print('Using BALLISTIC propagation.')
    # Final travel time from the converged distance to the BSN (km / km/s = s)
    shift = raw['X'].to_numpy() / velsmooth
    raw['tshift'] = shift

    # Arrival time at the boundary = original time + travel time
    times = pd.to_datetime(raw['time'])
    tshift = (times + pd.to_timedelta(shift, unit='s')).to_numpy()

    # Remove "overtaken" points (arrival time must increase monotonically)
    keep = [0]
    discard = []
    lasttime = tshift[0]
    for i in range(1, len(tshift)):
        if tshift[i] > lasttime:
            keep.append(i)
            lasttime = tshift[i]
        else:
            discard.append(i)
    print(f'Removing "overtaken" points {len(discard)} of {len(tshift)} total for event {e+1}.')


    #Puts all the raw data into Upstream Dataframe
    upstream = pd.DataFrame({
        'Time': raw['time'],
        'BX_GSM': raw['bx'],
        'BY_GSM': raw['by'],
        'BZ_GSM': raw['bz'],
        'VX':raw['ux'],
        'VY': raw['uy'],
        'VZ': raw['uz'],
        'N': raw['n'],
        'T': raw['t']

    }, columns=['Time', 'BX', 'BY', 'BZ','VX','VY','VZ','N','T']).replace(to_replace=[-1.00000E+31], value=np.nan)

    # Create new IMF object and populate with propagated values.
    # Use the information above to throw out overtaken points.
    path = os.path.join('./Plots/Outfile',f'{date_start.strftime('%Y-%m-%d')}_outfile.dat')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imfout = ImfInput(path, load=False, npoints=len(keep))
    for v in swmf_vars:
        imfout[v] = dmarray(raw[v][keep], {'units': units[v]})
    imfout['time'] = pd.to_datetime(tshift[keep]).to_pydatetime()  # if ImfInput expects datetime objects
    imfout.attrs['satxyz'] = [np.mean(bound_dist) / RE, 0, 0]

    downstream = pd.DataFrame({
        'Time': imfout['time'],
        'BX_GSM': imfout['bx'],
        'BY_GSM': imfout['by'],
        'BZ_GSM': imfout['bz'],
        'VX':imfout['ux'],
        'VY': imfout['uy'],
        'VZ': imfout['uz'],
        'N': imfout['n'],
        'T': imfout['t']

    }, columns=['Time', 'BX_GSM', 'BY_GSM', 'BZ_GSM','VX','VY','VZ','N','T']).replace(to_replace=[-1.00000E+31], value=np.nan)
    downstream = downstream.set_index('Time')
    downstream = downstream.resample('min').mean().interpolate(method='linear').ffill().bfill()
    downstream = downstream.reset_index()

    folder_path = f'./Data/Downstream/Ballistic'
    os.makedirs(folder_path, exist_ok=True)
    downstream.to_csv(f'{folder_path}/{date_string}.csv', index=False)


    #imfout.attrs['header'].append(f'Source data: {file1}\n')

    imfout.attrs['header'].append('Ballistically propagated from L1 to ' +
                                      'upstream BATS-R-US boundary\n')
    imfout.attrs['header'].append('\n')
    imfout.attrs['coor'] = 'GSM'
    imfout.attrs['satxyz'] = [np.mean(raw['X'])/RE, 0, 0]
    imfout.attrs['header'].append(f'File created on {datetime.now()}')

    imfout.write()

    # Plot!
    fig, ax = plt.subplots(5, 1, figsize=(8, 10))
    #fig = imfout.quicklook(['by', 'bz', 'n', 't', 'ux'])
    plotvars = ['by', 'bz', 'n', 't', 'ux']
    color = ['C0', 'C1', 'C2', 'C3', 'C4']
    timerange = [raw['time'][0], imfout['time'][-1]]

    for ax, v, c in zip(fig.axes, plotvars, color):
        #c = ax.get_lines()[0].get_color()
        ax.plot(imfout['time'], imfout[v], c=c)
        ax.plot(raw['time'], raw[v], '--', c=c, alpha=.5)
        #ax.plot(raw['time'][:][discard], raw[v][...][discard],
                #'.', c='crimson', alpha=.5)
    l1 = Line2D([], [], color='red', lw=2,
                label='Timeshifted (Downstream) Values')
    l2 = Line2D([], [], color='red', alpha=.5, linestyle='--', lw=2,
                label='Original (Reconstructed Upstream) Values')
    fig.legend(handles=[l1, l2], loc='upper center', ncol=2)
    #fig.subplots_adjust(top=.933)
    fig.subplots_adjust(hspace=0.04, top=0.95, right=0.95,
                        bottom=0.05 + 0.03)
    plt.suptitle('Ballistically-Propagated Solar Wind (With Upstream Source)')
    applySmartTimeTicks(ax, timerange, dolabel=ax == fig.axes[-1])
    #plt.xlabel('Time UTC')
    path = os.path.join('./Plots/OutputProp',f'{date_start.strftime('%Y-%m-%d')}outputprop.png')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)

    print(f'Finished processing event {e + 1}', flush=True)


def prefetch(e):
    print(f'Prefetching event {e + 1} of {len(event_list)}', flush=True)
    start = pd.to_datetime(event_list['Date_Start'][e])
    stop = pd.to_datetime(event_list['Date_Stop'][e])
    pyspedas.projects.omni.data(trange=[(start - timedelta(minutes=180)).strftime(FMT),
                                        stop.strftime(FMT)],
                                datatype='1min', level='hro2', time_clip=True)


if __name__ == '__main__':
    for e in range(len(event_list)):      # serial download pass, avoids file races
        prefetch(e)
    with Pool(processes=min(32, len(event_list)), maxtasksperchild=1) as pool:
        pool.map(process_event, range(len(event_list)))