import os
import requests
import zipfile
import numpy as np
from inspect import getsourcefile
import pandas as pd

from open.helper import get_continental_extent

rootdir = os.path.dirname(os.path.abspath(
    getsourcefile(lambda: 0))).split('\\src')[0]
datadir = os.path.join(rootdir, 'data_wg3')
constants_folder = os.path.join(rootdir, 'src', 'GHM_downscaling', 'constants_wg3')

#Create arcid
continentlist = ['eu']
xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
aoi = ((xmin, xmax), (ymin, ymax))
res = '5min'

write_raster_specs = {
    '30min': 2,
    '5min': 12,
    '6min': 10,
    '30sec': 120,
    '15s': 240
}

rmulti = write_raster_specs[res]
# Get number of cols and rows based on extent and conversion factor
no_cols = round((aoi[0][1] - aoi[0][0]) * rmulti)
no_rows = round((aoi[1][1] - aoi[1][0]) * rmulti)
cellsize = 1 / rmulti
xyarcid_x = np.arange(aoi[0][0]+(cellsize/2.), aoi[0][1]+(cellsize/2.), cellsize).reshape((1,no_cols))
xyarcid_y = np.arange(aoi[1][0]+(cellsize/2.), aoi[1][1]+(cellsize/2.), cellsize).reshape((no_rows, 1))

xyarcid_pd = pd.DataFrame(
    data={'X': np.repeat(xyarcid_x, no_rows, 0).flatten(),
          'Y': np.repeat(xyarcid_y, no_cols, 1).flatten(),
          'arcid': np.arange(no_cols*no_rows, dtype=np.int32)+1,
          'col_ix': np.repeat((np.arange(no_cols)+1).reshape(1,no_cols), no_rows, 0).flatten(),
          'row_ix': np.repeat((np.arange(no_rows)+1).reshape(no_rows,1), no_cols, 1).flatten()}
)

xyarcid_pd.to_csv(os.path.join(constants_folder, 'xyarcid.csv'),
                  index=False)


# #Download reference grid from HydroSHEDS
# DDM5_url = "https://data.hydrosheds.org/file/hydrosheds-v1-dir/hyd_glo_dir_5m.zip"
# zip_path_DDM5 = os.path.join(datadir, os.path.split(DDM5_url)[1])
#
# if not os.path.exists(zip_path_DDM5):
#     with open(zip_path_DDM5, "wb") as file:
#         # get request
#         print(f"Downloading DDM5")
#         response = requests.get(DDM5_url, verify=False)
#         file.write(response.content)
# else:
#     print(zip_path_DDM5, "already exists... Skipping download.")
#
# with zipfile.ZipFile(zip_path_DDM5, 'r') as zip_ref:
#     zip_ref.extractall(os.path.dirname(zip_path_DDM5))
#
# #Extract it



