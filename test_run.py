#RunDownscaling.py #####################################################################################################

import os
from inspect import getsourcefile
import pandas as pd
from codetiming import Timer

from open.DryverDownscalingWrapper import DryverDownscalingWrapper, gather_finished_downscaling, run_prepared_downscaling
from open.DryverDownscalingConfig import DownscalingConfig
from open.helper import get_continental_extent

rootdir = os.path.dirname(os.path.abspath(
    getsourcefile(lambda: 0))).split('\\src')[0]
localdir = os.path.join(rootdir, 'results', 'downscaling_output')
if not os.path.exists(localdir):
    os.mkdir(localdir)

continentlist = ['eu']  # ['eu', 'as', 'si']
continent = ''.join(continentlist)
wginpath = os.path.join(rootdir, 'data', 'WG_inout_downscaling_data',
                        'wghm22e_v001', 'input')  # '/home/home1/gm/datasets/input_routing/wghm22e_v001/input/'
wgpath = os.path.join(rootdir, 'data', 'WG_inout_downscaling_data', '22eant')  # '/home/home8/dryver/22eant/'
setup_folder = os.path.join(rootdir, 'data',
                            'setupdata_for_downscaling')  # '/home/home1/gm/projects/DRYvER/03_data/12_downscalingdata_eu/'
stations_path = os.path.join(setup_folder, 'stations.csv')
constants_folder = os.path.join(rootdir, 'src', 'DRYVER-main', 'constants')
pois = pd.read_csv(stations_path)  # points of interest
if continent in {'eu', 'as', 'si', 'sa'}:
    xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
    aoi = ((xmin, xmax), (ymin, ymax))
if continent == 'rhone':
    aoi = ((3.5, 9), (43, 48.5))

dconfig = DownscalingConfig(wg_in_path=wginpath,
                            wg_out_path=wgpath,
                            hydrosheds_path=setup_folder,
                            startyear=1901,
                            endyear=2019,
                            temp_dir=localdir,
                            write_raster=False,
                            write_result='nc',
                            mode='ts',
                            continent=continent,
                            constants_folder=constants_folder,
                            pois=pois,
                            runoff_src='srplusgwr',
                            correct_global_lakes=True,
                            sr_smoothing=False,
                            l12harm=False,
                            dis_corr=True,
                            large_river_corr=True,
                            corr_grid_shift=True,
                            corr_grid_smoothing=False,
                            correction_threshold_per_skm=0.001,
                            area_of_interest=aoi,
                            # corrweightfactor=0.1
                            )

#DryverDownscalingWrapper ##############################################################################################
import pickle
import glob
from multiprocessing import Pool
from functools import partial
import os

import numpy as np
import pandas as pd
import osgeo.gdal
from codetiming import Timer

from open.WGData import WGData
from open.HydroSHEDSData import HydroSHEDSData
from open.DownScaleArray import DownScaleArray
from open.DryverDownscaling import run_task

config=dconfig #################### ADDED
kwargs = dict()

with open(os.path.join(config.temp_dir, 'run_information.txt'), 'w') as f:
    temp = vars(config)
    for item in temp:
        f.write('{} : {}\n'.format(item, temp[item]))

mode = config.mode
dconfig = config
kwargs.update(config.kwargs)
kwargs = kwargs
wg = WGData(config=dconfig, **kwargs) #Get WaterGAP object instance (data and tools related to WG)
hydrosheds = HydroSHEDSData(dconfig, **kwargs) #Get WaterGAP object instance (data and tools related to HydroSHEDS)
wg.calc_continentalarea_to_landarea_conversion_factor() #Compute conversion factor to concentrate runoff from continental area contained in WG pixels to actual land area
wg.calc_surface_runoff_land_mm() #Apply conversion factor
if dconfig.mode == 'longterm_avg':
    wg.get_longterm_avg_version()
    wg.longterm_avg_converted = True
daysinmonth_dict = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

temp_downscalearray = DownScaleArray(config, dconfig.aoi, **kwargs)
surfacerunoff_based_dis = None
corrected_dis = None
correction_weights_15s = None
correction_grid_30min = None

