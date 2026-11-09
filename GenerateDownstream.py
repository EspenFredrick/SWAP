from argparse import ArgumentParser, RawDescriptionHelpFormatter
from multiprocessing.resource_sharer import stop
import pandas as pd
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
from dateutil.parser import isoparse

from Functions import functions as f
from pytplot import get_data

import os

from Functions import GetSatellitesGSM as GS
import pytz
from pyspedas import tplot

from numpy import bool_
from numpy.ma import MaskedArray
from scipy.interpolate import interp1d
from matplotlib.dates import date2num
from spacepy.plot import style






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




# Import the OMNI, ACE, and Wind data using the list of times
event_list = pd.read_csv(args.input, delimiter=',', header=0)
for e in range(len(event_list)):

    start = event_list['Date_Start'][e]
    date_stop = event_list['Date_Stop'][e]
    date_stop = pd.to_datetime(date_stop)
    date_start = pd.to_datetime(start)

    date_string = date_start.strftime('%Y-%m-%d')
    folder_path = f'./Data/Raw'
    date_string = date_start.strftime('%Y-%m-%d')
    raw = pd.read_csv(f'{folder_path}/{date_string}.csv',parse_dates=['time'])

    # Get S/C distance. If not in file, use approximation.
    print(f'Run {e+1} out of {len(event_list)}.')
    pyspedas.projects.omni.data(trange=[(date_start - timedelta(minutes=180)).strftime(FMT), date_stop.strftime(FMT)],
                                    datatype='1min', level='hro2', time_clip=True)
    t = get('BZ_GSM')[0]
    omni = GS.regrid(t, {'Time':pd.to_datetime(get('BZ_GSM')[0],unit='s'),'BY_GSM': get('BY_GSM')[1], 'BZ_GSM': get('BZ_GSM')[1],
                             'IMF': get('IMF')[1], 'ts': get('Timeshift')[1],
                             'VX': get('Vx')[1], 'VY': get('Vy')[1], 'VZ': get('Vz')[1],
                             'N': get('proton_density')[1], 'T': get('T')[1],
                             'XPOS': get('x')[1], 'BSN': get('BSN_x')[1]}, 'min')
    print(omni['Time'][0])
    if args.bs == 'dynamic':
        bound_dist = np.average(omni['BSN'])*RE
    else:
        bound_dist = 32 * RE  # Default distance to BATS-R-US upstream boundary from Earth.
    max_iterations = 10
    tolerance = 0.05 * RE  # Stop looping if BSN shifts by less than 0.05 Earth Radii
    for iteration in range(max_iterations):
         # 1. Calculate space-craft distance to boundary based on current bound_dist
        if 'pos' in raw:
            print('S/C location found! Using dynamic location.')
            raw['X'] = raw['pos'] - bound_dist
        else:
            print('S/C location NOT found, using static L1 distance.')
            raw['X'] = l1_dist - bound_dist
            
        # 2. Compute time shift (Ensure raw['ux'] is provided in km/s or converted to match distance units!)
        raw['tshift'] = raw['X'] / raw['ux']
        shifted_time = pd.to_datetime(raw['time']) + pd.to_timedelta(raw['tshift'], unit='s')
        shifted_time_min = shifted_time.dt.ceil('min')

        print(shifted_time_min)

        loc = pd.Index(omni['Time']).get_indexer(shifted_time_min)
        loc = np.clip(loc, 0, len(omni) - 1)
        print(loc)
        if args.bs != 'dynamic':
            # Static boundary doesn't need to loop/converge
            break
                # 5. Extract the dynamic BSN from OMNI at those specific propagation times
        new_bound_dist = omni['BSN'].iloc[loc].values * RE
        
        # 6. Check for convergence: Has the boundary distance stabilized?
        current_bound_dist = np.atleast_1d(bound_dist)
        change = np.max(np.abs(new_bound_dist - current_bound_dist))
        
        # Update bound_dist array for the next iteration
        bound_dist = new_bound_dist
        
        print(f"Iteration {iteration+1}: Max BSN change balance is {change / RE:.3f} RE")
        if change < tolerance:
            print("Convergence achieved!")
            break
    else:
        print("Warning: Reached max iterations without perfect convergence.")
        
            # Apply velocity smoothing as required
   # Create seconds-from-start time array:
    tsec = np.array([(t - raw['time'][0]).total_seconds() for t in raw['time']])

    # Apply velocity smoothing as required
    print(f'Applying smoothing using a 1 window size.')
    velsmooth = medfilt(raw['ux']*RE, 1)

    print('Using BALLISTIC propagation.')
    shift = raw['X']/velsmooth  # Time shift per point.
    # Apply shift to times.
    tshift = np.array([t1 - timedelta(seconds=t2) for t1, t2 in
                       zip(raw['time'], raw['tshift'])])

    # Ensure that any points that are "overtaken" (i.e., slow wind overcome by
    # fast wind) are removed. First, locate those points:
    keep = [0]
    discard = []
    lasttime = tshift[0]
    for i in range(1, len(raw['time'])):
        if tshift[i] > lasttime:
            keep.append(i)
            lasttime = tshift[i]
        else:
            discard.append(i)
    print(f'Removing "overtaken" points {len(discard)} of {tsec.size} total.')


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
    imfout['time'] = tshift[keep]

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

    folder_path = f'./Data/Downstream'
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