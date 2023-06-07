import numpy as np
import pandas as pd


def read_obs_station(filename, lat_speedy, lon_speedy):
    """
    Read the obs stations (*.tbl)

    Args:
        filename (str): input obs list
        lat_speedy (array): latitude in Speedy (not uniform)
        lon_speedy (array): longitude in Speedy

    Returs:
        lat (array): latitude coordinates in map
        lon (array): longitude coordindate in map
        row (int): number of obs stations
    """
    f = pd.read_csv(filename, sep=r"\s+", index_col=None,
                    header=None, skiprows=2)
    row, _ = f.shape
    lat, lon = np.zeros(row), np.zeros(row)
    for i in range(row):
        lat[i] = lat_speedy[int(f.iloc[i][1] - 1)]
        lon[i] = lon_speedy[int(f.iloc[i][0] - 1)]
    return lat, lon, row
