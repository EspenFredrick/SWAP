# Place all downloaded pySPEDAS data in the /Data/Raw/ folder
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['SPEDAS_DATA_DIR'] = 'Data/Raw'

# Import statements for various libraries required
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import numpy as np
import pandas as pd
import pyspedas
from pytplot import get_data
from multiprocessing import Pool

from Functions import GetSatellitesGSM as GS

# Declare important constants. These will be used for manipulating positions/temperatures/etc.
KB = 1.380649E-23                  # Boltzmann constant [J/K]
MP = 1.67262192E-27                # proton mass [kg]
FILL = -1e30                       # ACE/Wind fill values are -1e31
L1_LEAD = pd.Timedelta(hours=2)    # ACE/Wind start earlier so there is data to propagate
RE = 6371              # Earth radius in kilometers
l1_dist = 1495980      # L1 distance in km
bound_dist = 32 * RE   # Distance to BATS-R-US upstream boundary from Earth
FMT = '%Y-%m-%d %H:%M:%S'

# Command line arguments
parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
parser.add_argument('input', type=str, help='CSV file listing the events to download.')
parser.add_argument("--set_downstream", default=True, action='store_true', help='The --set_downstream flag takes True or False. '
                    'True takes the average BSN distance and propagates the data there. False propagates it to the BATS-R-US upstream boundary')
parser.add_argument('--force', action='store_true',
                    help='Redownload events even if their output files already exist.')
args = parser.parse_args()

# Function that returns just the (times, values) columns of a loaded pytplot variable. Used to simplify things
def get(name):
    d = get_data(name)
    return d[0], d[1]

# Function that prints what is going on in the terminal, so you know that the program isn't hanging
# It just prints the message you put in (msg) followed by a new line and 52 tildes (~)
def say(msg):
    print(f'{msg}\n{"~" * 52}')

events = pd.read_csv(args.input)   # module level, so workers see it
FILES = ['omni', 'artemis', 'ace_mfi', 'ace_swe', 'wind_mfi', 'wind_swe']

def download_event(i):
    ev = events.iloc[i]
    start, stop = pd.to_datetime(ev['Date_Start']), pd.to_datetime(ev['Date_Stop'])
    folder = f'./Data/Interpolated/{start:%Y-%m-%d}'

    if not args.force and all(os.path.exists(f'{folder}/{n}.csv') for n in FILES):
        print(f'Skipping event {i + 1}: already downloaded', flush=True)
        return

    l1 = [(start - L1_LEAD).strftime(FMT), stop.strftime(FMT)]
    print(f'Event {i + 1}: {start} to {stop}', flush=True)

    # OMNI (already propagated to the bow shock nose)
    say('Downloading OMNI')
    pyspedas.projects.omni.data(trange=[start.strftime(FMT), stop.strftime(FMT)],
                                datatype='1min', level='hro2', time_clip=True)
    t = get('BZ_GSM')[0]
    # regrid is a function in GetSatellitesGSM that interpolates.
    omni = GS.regrid(t, {'BY_GSM': get('BY_GSM')[1], 'BZ_GSM': get('BZ_GSM')[1],
                         'IMF': get('IMF')[1], 'ts': get('Timeshift')[1],
                         'VX': get('Vx')[1], 'VY': get('Vy')[1], 'VZ': get('Vz')[1],
                         'N': get('proton_density')[1], 'T': get('T')[1],
                         'XPOS': get('x')[1], 'BSN': get('BSN_x')[1]}, 'min')

    # ARTEMIS
    say('Downloading ARTEMIS')
    artemis = GS.get_artemis_gsm(start, stop)

    # ACE
    say('Downloading ACE')
    pyspedas.projects.ace.mfi(trange=l1, datatype='h3', time_clip=True)
    t, b = get('BGSM')
    ace_mfi = GS.regrid(t, {'BX': b[:, 0], 'BY': b[:, 1], 'BZ': b[:, 2]}, 's', FILL)

    pyspedas.projects.ace.swe(trange=l1, datatype='h0', time_clip=True)
    t, v = get('V_GSE')
    ace_swe = GS.regrid(t, {'VX': v[:, 0], 'VY': v[:, 1], 'VZ': v[:, 2],
                            'NP': get('Np')[1], 'Temp': get('Tpr')[1]}, 's', FILL)

    # Wind
    say('Downloading Wind')
    pyspedas.projects.wind.mfi(trange=l1, datatype='h0', time_clip=True)
    t, b = get('B3GSM')
    wind_mfi = GS.regrid(t, {'BX': b[:, 0], 'BY': b[:, 1], 'BZ': b[:, 2]}, 's', FILL)

    pyspedas.projects.wind.swe(trange=l1, datatype='h1', time_clip=True)
    w = get('Proton_W_moment')[1]
    w = np.where(w <= FILL, np.nan, w)  # mask fill before converting to K
    temp = MP / (2 * KB) * (w * 1000) ** 2  # thermal speed [km/s] -> K
    t, vx = get('Proton_VX_moment')
    wind_swe = GS.regrid(t, {'VX': vx, 'VY': get('Proton_VY_moment')[1],
                             'VZ': get('Proton_VZ_moment')[1],
                             'NP': get('Proton_Np_moment')[1], 'Temp': temp}, 's', FILL)

    # Save
    os.makedirs(folder, exist_ok=True)
    for name, df in dict(omni=omni, artemis=artemis, ace_mfi=ace_mfi, ace_swe=ace_swe,
                         wind_mfi=wind_mfi, wind_swe=wind_swe).items():
        df.to_csv(f'{folder}/{name}.csv')  # Time index is written as a column

    print(f'Finished event {i + 1}', flush=True)


def safe_download(i):
    try:
        download_event(i)
    except Exception as ex:
        print(f'FAILED event {i + 1}: {ex!r}', flush=True)

if __name__ == '__main__':
    with Pool(processes=min(6, len(events)), maxtasksperchild=1) as pool:
        pool.map(safe_download, range(len(events)))