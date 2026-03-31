import numpy as np
import spacepy

def gse_to_gsm(x, y, z, time):
    '''
    Use spacepy's Coord module to rotate from GSE to GSM.

    Parameters
    ----------
    x, y, z : Numpy array-like
        The GSE X, Y, and Z values of the timeseries to rotate.
    time : array of Datetimes
        The time corresponding to the xyz timeseries.

    Returns
    -------
    x, y, z : Numpy arrays
        The rotated values now in GSM coordinates.
    '''

    from spacepy.coordinates import Coords
    from spacepy.time import Ticktock
    print(type(time[1]))
    print('~~~~~~~~~~~~~~~~~~~~~~~~~')
    print(time[0])
    print(time[1])
    # Convert time to ticktocks:
    ticks = Ticktock(time, 'ISO')

    # Rearrange values to correct shape:
    xyz = np.array([x, y, z]).transpose()

    # Rotate:
    gse_vals = Coords(xyz, 'GSE', 'car', ticks=ticks)
    gsm_vals = gse_vals.convert('GSM', 'car')

    return gsm_vals.x, gsm_vals.y, gsm_vals.z


def unify_time(time1, time2):
    '''
    Given two timeseries, combine all unique points into a single array.
    '''

    time1 = time1.tolist()
    time2 = time2.tolist()

    for t in time2:
        if t not in time1:
            time1.append(t)

    time1.sort()
    return np.array(time1)

def pair(time1, data, time2, varname=None, **kwargs):
    '''
    Use linear interpolation to pair two timeseries of data.  Data set 1
    (data) with time t1 will be interpolated to match time set 2 (t2).
    The returned values, d3, will be data set 1 at time 2.
    No extrapolation will be done; t2's boundaries should encompass those of
    t1.

    Bad data values will be removed prior to interpolation. The `varname`
    kwarg will determine what is considered "bad". Possible varnames include:
    n, t, u[xyz], b[xyz], pos

    This function will correctly handle masked functions such that masked
    values will not be considered.

    **kwargs** will be handed to scipy.interpolate.interp1d
    A common option is to set fill_value='extrapolate' to prevent
    bounds errors.
    '''

    from numpy import bool_
    from numpy.ma import MaskedArray
    from scipy.interpolate import interp1d
    from matplotlib.dates import date2num

    # Dates to floats:
    t1 = date2num(time1)
    t2 = date2num(time2)

    # Search for bad data values and remove:
    loc = ~np.isfinite(data) | (np.abs(data) >= 1E15)
    if varname in ['n', 'alpha']:
        loc = (loc) | (data > 1E4)
    elif varname == 't':
        loc = (loc) | (data > 1E7)
    elif varname in ['ux', 'uy', 'uz']:
        loc = (loc) | (np.abs(data) > 1E4)

    t1, data = t1[~loc], data[~loc]

    # Remove masked values (if given):
    if type(data) is MaskedArray:
        if type(data.mask) is not bool_:
            d = data[~data.mask]
            t1 = t1[~data.mask]
        else:
            d = data
    else:
        d = data

    # Create interpolator function
    func = interp1d(t1, d, fill_value='extrapolate', **kwargs)

    # Interpolate and return.
    return func(t2)