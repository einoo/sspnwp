# E-optimizaiton of sparse sensor placement
import argparse
import os
from itertools import product

import numpy as np

parser = argparse.ArgumentParser(
    description="Search in the RIO\
for the maximum of spread at each time step"
)
parser.add_argument(
    "--location",
    "-l",
    type=str,
    dest="location",
    required=True,
    help="Directory of ensemble state location",
)
parser.add_argument(
    "--time",
    "-t",
    type=str,
    dest="time",
    required=True,
    help="Time of the current step",
)
parser.add_argument(
    "--radius", "-r", type=int, dest="lr", required=True, help="Radius of localization"
)
parser.add_argument(
    "--number",
    "-n",
    type=int,
    dest="obsn",
    default=1,
    required=True,
    help="Number of additional obs",
)
parser.add_argument(
    "--ensemble", "-e", type=int, dest="enm", required=True, help="Ensemble members"
)
args = parser.parse_args()

NLON = 96  # Number of grids in lontitude
NLAT = 48  # Number of grids in latitude
REC = 18  # 4th layer of temperature
ENM = args.enm  # Ensemble members

lat_nh = [
    87.159,
    83.479,
    79.777,
    76.070,
    72.362,
    68.652,
    64.942,
    61.232,
    57.521,
    53.810,
    50.099,
    46.389,
    42.678,
    38.967,
    35.256,
    31.545,
    27.833,
    24.122,
    20.411,
    16.700,
    12.989,
    9.278,
    5.567,
    1.856,
]
lat_speedy = [-i for i in lat_nh]
lat_nh.reverse()
lat_speedy.extend(lat_nh)
lon_speedy = (np.linspace(0, 360, num=NLON, endpoint=False),)
lon_speedy = lon_speedy[0].tolist()


# data-driven sparse sensor placement ###
def main():
    r, p = ENM, args.obsn  # r :: num of mode, p :: num of sensor
    # 0. prepare training data like ensemble perturbations
    # read observation stations within the cutoff range
    filename = os.path.join("./", "station_roi_%d.tbl" % (args.lr))
    lon_roi, lat_roi, _ = read_obs_station(filename)
    # find existing observations in raob
    raobfile = os.path.join("./", "station_raob.tbl")
    lon_raob, lat_raob, _ = read_obs_station(raobfile)
    replicate = np.zeros(len(lon_roi), dtype=int)
    for g, h in zip(lon_raob, lat_raob):
        z = -1
        for j, k in zip(lon_roi, lat_roi):
            z += 1
            if j == g and k == h:
                replicate[z] = 1
    # read the ensemble states from the previous step
    X = []
    for i in range(ENM):
        #  state = read_binary_data(args.location, NLON, NLAT, REC)
        state_file = os.path.join(
            args.location, "%06d" % (i + 1), "%s.grd" % (args.time)
        )
        state = read_binary_data(state_file, NLON, NLAT, REC)
        # obtain the state variables within ROI, return 1D array
        x = reshape_state(lat_roi, lon_roi, state)
        X.append(x)
    X = np.array(X)
    X = X.T
    xmean = np.mean(X, axis=1)
    DX = np.array([r - xmean for r in X.T])
    DX = DX.T
    # 1. culculate left singular vector u of training data
    u, _, _ = np.linalg.svd(DX, full_matrices=False)  # dimension reduction
    # 2. sensor placement (QR method)
    H = SensorPlacement(u, r, p, replicate)
    # 3. obtain the sensor location in speedy coordinates
    ssp_loc = sensor_location(lon_roi, lat_roi, H)
    for i in range(p):
        print(
            "  > ENSEMBLE EHMODE DETERMINED OBS LOCATION IS: %d, %d."
            % (ssp_loc[i][0], ssp_loc[i][1])
        )
    filename_max_coord = os.path.join(
        ".", "max_coord_ehmode_%d.tbl" % (args.lr))
    ssp_x = [row[0] for row in ssp_loc]
    ssp_y = [row[1] for row in ssp_loc]
    save_obs_station(filename_max_coord, ssp_x, ssp_y)


# EG method
def SensorPlacement(u, r, p, eobs):
    u = u[:, 0:r]
    n, r = u.shape
    # calculate the replicate index
    exist = np.where(eobs == 1)
    ep = np.sum(eobs)
    tp = p + ep  # total observations in the candidate region
    print(ep, p, tp)
    H = np.zeros((tp, n), dtype=int)
    for i in range(len(exist[0])):
        H[i, exist[0][i]] = 1
    S = []
    C = np.zeros((tp, r))
    CT = H @ u
    C[:ep] = CT[:ep]
    CC = C[: ep + 1]
    for i in range(len(exist[0])):
        u[exist[0][i]] = 0.0
    for obs_num in range(p):
        obj = np.zeros(n)
        if obs_num + ep < r:
            for obs_loc in range(n):
                CC[-1] = u[obs_loc]
                obj[obs_loc] = np.min(np.linalg.eig(CC @ CC.T)[0])
        else:
            for obs_loc in range(n):
                CC[-1] = u[obs_loc]
                obj[obs_loc] = np.min(np.linalg.eig(CC.T @ CC)[0])
        # Obtain the sensor location from sensor candidate matrix
        S.append(np.argmax(obj))
        H[obs_num + ep, np.argmax(obj)] = 1.0
        CT = H @ u
        C[obs_num + ep] = CT[obs_num + ep]
        CC = C[: obs_num + ep + 2]
        # Set the found location to zero to excluded from candidate
        u[np.argmax(obj)] = 0.0

    return S


def read_binary_data(filename, nlon, nlat, rec):
    """
    Read the fortran unformatted binary file to numpy array

    Args:
        filename (str): The name of the grd file
        nlon (int):     The number in longitude
        nlat (int):     The number in latitude
        ret (int):      The number of record

    Returns:
        pdata (array):  In dimension (nlon, nlat) of float32
    """
    dstart = nlon * nlat * (rec - 1)

    pdata = np.zeros((nlat, nlon), dtype=">f")
    with open(filename, mode="rb") as bin_f:
        data = np.fromfile(bin_f, dtype=">f")
        for i, j in product(range(nlat), range(nlon)):
            pdata[i, j] = data[dstart + i * nlon + j]
    return pdata


def read_obs_station(filename):
    """
    Read the observation stations with regular interval

    Args:
        filename (str): Input filename for observation coordinates

    Return:
        lat (list): List of latitude coordinates
        lon (list): List of longitude coordinates
        row (int):  Total number of stations
    """
    with open(filename, "r") as f:
        lines = f.readlines()
    row = len(lines) - 2
    lat, lon = np.zeros(row, dtype=int), np.zeros(row, dtype=int)
    for i in range(row):
        lat[i] = int(lines[i + 2].split()[1]) - 1
        lon[i] = int(lines[i + 2].split()[0]) - 1
    return lon, lat, row


def reshape_state(roi_xcoords, roi_ycoords, state_data):
    """
    Return the states in ROI, and convert to x array

    Args:
        roi_xcoords (list):  list of ROI lat coordinates
        roi_ycoords (list):  list of ROI lon coordinates
        state_data (array): 2D array of state data

    Return:
        state_x (array): array of state variables
    """
    state_x = np.zeros(len(roi_xcoords))
    for i in range(len(roi_xcoords)):
        x = roi_xcoords[i]
        y = roi_ycoords[i]
        state_x[i] = state_data[x, y]
    return state_x


def sensor_location(lat, lon, h):
    """
    Obtain the sensor location in ROI

    Args:
        lat, lon (list): observation coordinate candidates
        h (list): The number of coordinate order

    Return:
        ssp_loc (list): The location of the sensor placement
    """
    ssp_loc = []
    for i in h:
        ssp_loc.append([lat[i], lon[i]])
    return ssp_loc


def save_obs_station(filename, xcoord, ycoord):
    """
    Save the observation stations

    Args:
        filename (str): Output filename for coordinates storage
        xcoord (list):  List of latitude
        ycoord (list):  List of longitude

    Return:
        None
    """
    if os.path.exists(filename):
        os.remove(filename)
        print("  > DELETE THE EXISTING OBSERVATIONS IN ROI")
    f = open(filename, "w")
    f.write("  I  J\n")
    f.write("------\n")
    # Index to be consistent with the fortran generated data
    for i in range(len(xcoord)):
        f.write("%3d%3d\n" % (xcoord[i] + 1, ycoord[i] + 1))
    f.close()
    print("  > ADDITIONAL OBSERVATION COORDINATES WERE WRITTEN IN %s" % (filename))
    return None


if __name__ == "__main__":
    main()
