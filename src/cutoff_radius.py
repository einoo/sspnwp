from itertools import product

import numpy as np
from haversine import haversine


def cutoff_radius(center, scale_length):
    """
    Draw the localized cutoff radius

    Args:
        center (array): Center point of the interest
        scale_length (float): localization scale length

    Return:
        cutoff_lat (float): latitude coordinate of the circle
        cutoff_lon (float): longitude coordinate of the circle
    """
    radius = 2.0 * np.sqrt(10.0 / 3.0) * scale_length
    cutoff_lat = []
    cutoff_lon = []
    for i, j in product(np.arange(90, -90.1, -0.1), np.arange(-180.0, 180.0, 0.1)):
        length = haversine((i, j), (center[0], center[1]), unit="km")
        if abs(length - radius) <= 2:
            cutoff_lat.append(i)
            cutoff_lon.append(j)
    # rearrange the coordinates to draw a circle
    sep_lon1 = cutoff_lat.index(max(cutoff_lat))
    sep_lon2 = cutoff_lat.index(min(cutoff_lat))
    radi = np.zeros((len(cutoff_lon), 2))
    j = 0
    for i in range(len(cutoff_lon)):
        if cutoff_lon[i] <= min(cutoff_lon[sep_lon1], cutoff_lon[sep_lon2]):
            radi[j, :] = np.array([cutoff_lat[i], cutoff_lon[i]])
            j += 1
    for i in range(len(cutoff_lon) - 1, 0, -1):
        if cutoff_lon[i] > min(cutoff_lon[sep_lon1], cutoff_lon[sep_lon2]):
            radi[j, :] = np.array([cutoff_lat[i], cutoff_lon[i]])
            j += 1
    radi[-1, :] = radi[0, :]

    return radi[:, 0], radi[:, 1]
    # return cutoff_lat, cutoff_lon
