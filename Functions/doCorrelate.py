from scipy.stats import pearsonr
import numpy as np
import pandas as pd
import os

from spacepy.datamanager import values_to_steps


def prediction_efficiency_artemis(omni, artemis):  # This function just finds prediction efficiency
    return 1 - (np.sum((omni-artemis)**2)/np.sum((artemis - np.mean(artemis))**2))

def prediction_efficiency_omni(omni, artemis):  # This function just finds prediction efficiency
    return 1 - (np.sum((artemis-omni)**2)/np.sum((omni - np.mean(omni))**2))

def find_max_value_and_index(values):
    if all(c < 0 for c in values[1:]):
        return max(values[1:], key=abs), values.index(max(values[1:], key=abs))
    else:
        return max(values[1:]), values.index(max(values[1:]))

def correlate(file_pair, output_path, variable='BZ_GSM'):
    artemis, omni = file_pair   # Separate Artemis and Omni files

    num_windows = len(omni)-60  # Number of windows (60-min chunks of time)

    print(f'Computing {num_windows} windows...')

    data_rows = []  # Storage array to hold data
    offset_rows = []    # Storage array to hold time shift
    for n in range(num_windows):
        if omni['Time'].iloc[n + 59] <= artemis['Time'].iloc[-1]:

            artemis_start = artemis.loc[artemis['Time'] == omni['Time'].iloc[n]].index[0]   # Start the Artemis slice at the timestamp equal to the first timestamp of the OMNI slice
            artemis_stop = artemis.loc[artemis['Time'] == omni['Time'].iloc[n + 59]].index[0]   # Stop the Artemis slice at the last timestamp of the OMNI slice

            o_avg_xpos = np.average(omni['Xpos'][n:n + 59])     # Find the average x-position of OMNI over the interval in km
            a_avg_xpos = np.average(artemis['Xpos'][artemis_start:artemis_stop])    # Find the average x-position of Artemis over the interval in km
            hourly_offset = a_avg_xpos - o_avg_xpos     # Get their separation
            hourly_velocity = np.average(omni['VX'][n:n + 59])  # Get the x-axis SW velocity over the interval
            pred_arrival = int((hourly_offset/np.abs(hourly_velocity))/60)    # Calculate by ballistic shift how long it should take to traverse the distance

            metric_names = ['Pearson', 'PE_Artemis', 'PE_Omni']    # Names for the things we are solving for
            metric_stores = {name: [] for name in metric_names}     # Storage dictionary for metrics, create a key with the metric name and blank array as the value
            target_stores = {name + ' Target': [] for name in metric_names}     # Storage array for the target values we are trying to find (in this case, largest CC)

            for i in range(31):     # Sliding the Artemis array with respect to the fixed OMNI array 30 minutes
                o_slice = omni[variable][n:n + 59]  # Get the 60-minute OMNI chunk (i.e. :00 to :59)
                a_slice = artemis[variable][artemis_start - i:artemis_stop - i] # Get the 60-minute Artemis chunk (i.e. :00-:59, :01-:00, etc.)

                correlation, _ = pearsonr(o_slice, a_slice)    # Get a Pearson correlation for this interval
                pe_a = prediction_efficiency_artemis(o_slice, a_slice)    # Get the prediction efficiency
                pe_o = prediction_efficiency_omni(o_slice, a_slice)    # Get the prediction efficiency
                vars = (correlation, pe_a, pe_o)

                for count, (metric, store) in enumerate(metric_stores.items()):
                    store.append(vars[count])

            pearson_max_value, pearson_max_index = find_max_value_and_index(metric_stores['Pearson'])
            target_stores['Pearson Target'] = [pearson_max_value, pearson_max_index]
            target_stores['PE_Artemis Target'] = [metric_stores['PE_Artemis'][pearson_max_index], pearson_max_index]
            target_stores['PE_Omni Target'] = [metric_stores['PE_Omni'][pearson_max_index], pearson_max_index]


            data_rows.append(np.concatenate(([omni['Time'][n]], [omni['Time'][n + 59]], *[store[0] for name, store in target_stores.items()], hourly_offset, hourly_velocity, pred_arrival), axis=None))
            offset_rows.append(np.concatenate(([omni['Time'][n]], [omni['Time'][n + 60]], *[store[1] for name, store in target_stores.items()], pred_arrival), axis=None))

    values = pd.DataFrame(data_rows, columns=['Start', 'Stop', 'Pearson', 'PE_Artemis', 'PE_Omni', 'hourly-position', 'hourly-velocity', 'expected-arrival'])
    shifts = pd.DataFrame(offset_rows, columns=['Start', 'Stop', 'Pearson', 'PE_Artemis', 'PE_Omni', 'expected-arrival'])

    if not os.path.exists(os.path.join(output_path, f'{variable}/metrics/')) and not os.path.exists(os.path.join(output_path, f'{variable}/shifts/')):
        os.makedirs(os.path.join(output_path, f'{variable}/metrics/'), exist_ok=True)
        os.makedirs(os.path.join(output_path, f'{variable}/shifts/'), exist_ok=True)

    values.to_csv(os.path.join(output_path, f'{variable}/metrics/{artemis["Time"][0].strftime("%Y-%m-%d")}.csv'))
    shifts.to_csv(os.path.join(output_path, f'{variable}/shifts/{artemis["Time"][0].strftime("%Y-%m-%d")}.csv'))

