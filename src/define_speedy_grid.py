import numpy as np


def define_speedy_grid(nlat, nlon):
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
    assert len(lat_speedy) == nlat, "Latitude grids equals"
    lon_speedy = (np.linspace(-180, 180, num=nlon, endpoint=False),)
    lon_speedy = lon_speedy[0].tolist()
    return lat_speedy, lon_speedy
