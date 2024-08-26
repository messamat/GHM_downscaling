from datetime import datetime

#RunDownscaling.py #####################################################################################################
import os
from inspect import getsourcefile
from osgeo import gdal, osr, ogr
import pandas as pd
from codetiming import Timer

from open.DryverDownscalingWrapper import DryverDownscalingWrapper, gather_finished_downscaling, run_prepared_downscaling
from open.DryverDownscalingConfig import DownscalingConfig
from open.helper import get_continental_extent

rootdir = os.path.dirname(os.path.abspath(
    getsourcefile(lambda: 0))).split('\\src')[0]
localdir = os.path.join(rootdir, 'results', 'downscaling_output_{}'.format(datetime.today().strftime('%Y%m%d')))
if not os.path.exists(localdir):
    os.mkdir(localdir)

continentlist = ['eu']  # ['eu', 'as', 'si']
continent = ''.join(continentlist)
wginpath = os.path.join(rootdir, 'data', 'WG_inout_downscaling_data',
                        'wghm22e_v001', 'input')  # '/home/home1/gm/datasets/input_routing/wghm22e_v001/input/'
wgpath = os.path.join(rootdir, 'data', 'WG_inout_downscaling_data', '22eant')  # '/home/home8/dryver/22eant/'
hydrosheds_folder = os.path.join(rootdir, 'data',
                                 'hs_reproduced')  # '/home/home1/gm/projects/DRYvER/03_data/12_downscalingdata_eu/'
setup_folder = os.path.join(rootdir, 'data', 'setupdata_for_downscaling')
stations_path = os.path.join(setup_folder, 'stations.csv')
constants_folder = os.path.join(rootdir, 'src', 'GHM_downscaling', 'constants')
pois = pd.read_csv(stations_path)  # points of interest
if continent in {'eu', 'as', 'si', 'sa'}:
    xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
    aoi = ((xmin, xmax), (ymin, ymax))
if continent == 'rhone':
    aoi = ((3.5, 9), (43, 48.5))

dconfig = DownscalingConfig(wg_in_path=wginpath,
                            wg_out_path=wgpath,
                            hydrosheds_path=hydrosheds_folder,
                            startyear=2003,
                            endyear=2004,
                            temp_dir=localdir,
                            write_raster=True,
                            write_result='nc',
                            write_dir=localdir,
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

#--------------------------- Run class definition: down = DryverDownscalingWrapper(dconfig) ---------------------------------------------------------------------
with open(os.path.join(config.temp_dir, 'run_information.txt'), 'w') as f:
    temp = vars(config)
    for item in temp:
        f.write('{} : {}\n'.format(item, temp[item]))

mode = config.mode
dconfig = config
kwargs.update(config.kwargs)
kwargs = kwargs
wg = WGData(config=dconfig, **kwargs) #Get WaterGAP object instance (data and tools related to WG)
hydrosheds = HydroSHEDSData(dconfig, **kwargs) #Get HydroSHEDS object instance (data and tools related to HydroSHEDS)

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

#--------------------------- Run down.prepare() (L70 of RunDownscaling.py) ---------------------------------------------------------------------
staticdata = {
            'mean_land_fraction': wg.land_fractions.data.reset_index().groupby('arcid')['landareafr'].mean(),
            'wg_input': wg.wg_input.data,
            'coords': wg.coords,
            'flowacc': hydrosheds.flowacc,
            'landratio_corr': hydrosheds.get_wg_corresponding_grid(wg.landratio_corr_path),
            'largerivers_mask': hydrosheds.largerivers_mask(),
            'cell_pourpixel': hydrosheds.get_cell_pourpixel(),
            '30mingap_flowacc': hydrosheds.get_wg_corresponding_grid(wg.gap_flowacc_path),
            'keepgrid': hydrosheds.keepGrid.copy(),
            'shiftgrid': hydrosheds.shiftGrid.copy(),
            'upstream_pixelarea': hydrosheds.upa,
            'hydrosheds_geotrans': hydrosheds.hydrosheds_geotrans,
            'pixelarea': hydrosheds.pixarea,
            'globallakes_fraction_15s': hydrosheds.globallakes_fraction_15s_ar
        }

with open(os.path.join(dconfig.temp_dir, 'staticdata.pickle'), 'wb') as f:
    pickle.dump(staticdata, f)

# Create a list that holds, for each year-month the time-series data that are required to run the downscaling
# Each of these time steps thus represent a discrete "task"
taskdata_dict = {}
timestep=1
for yr in range(dconfig.startyear, dconfig.endyear + 1):
        for mon in range(1, 13):
            print(timestep)
            taskdata_dict["{0}_{1}".format(yr, mon)] = {
                'i': timestep,
                'sr': wg.surface_runoff_land_mm.data.set_index(['arcid',
                                                                'month',
                                                                'year'])['variable'].loc[
                    slice(None), mon, yr],
                'netdis_30min_series': wg.cell_runoff.set_index(['arcid',
                                                                 'month',
                                                                 'year'])['net_cell_runoff'].loc[
                    slice(None),
                    mon, yr],
                'dis': wg.dis.data.set_index(['arcid', 'month', 'year'])['dis'].loc[
                    slice(None), mon, yr],
                'gwrunoff': wg.gw_runoff.data.set_index(['arcid', 'month', 'year'])['variable'].loc[
                    slice(None), mon, yr],
                'month': mon,
                'year': yr,
                'totalrunoff': wg.total_runoff.data.set_index(['arcid',
                                                               'month',
                                                               'year'])['variable'].loc[
                    slice(None), mon, yr],
            }
            if dconfig.correct_global_lakes:
                taskdata_dict["{0}_{1}".format(yr, mon)]['globallakes_addition'] = (
                    wg.globallakes_addition.loc)[slice(None), yr, mon]
            timestep += 1

# Write list of tasks to pickle
for t in taskdata_dict:
    out_pickle = os.path.join(dconfig.temp_dir,
                              'data_task{:03d}.pickle'.format(taskdata_dict[t]['i'])
                              )
    with open(out_pickle, 'wb') as f:
        pickle.dump(taskdata_dict[t], f)

if config:
    dconfig.pickle()

#4.######################### Run DryverDownscaling.run_task() > save_and_run_ts > run_ts for year=2003, month=08 ###############################
import warnings
import argparse
import glob
from functools import partial
import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal, ogr, osr
import netCDF4
import numba
from numba import cfunc, carray
from numba.types import intc, CPointer, float64, intp, voidptr
from scipy.ndimage import generic_filter
from scipy import LowLevelCallable

from open.DownstreamGrid import get_inflow_sum, get_downstream_grid
from open.ModifiedFlowAcc import FlowAccTT
from open.DownScaleArray import DownScaleArray

year=2003
month=8
#Compute total runoff
#elif self.dconfig.runoff_src == 'srplusgwr':
srplusgwr = taskdata_dict['2003_8']['sr'] + taskdata_dict['2003_8']['gwrunoff']
#self.dconfig.sr_smoothing == False
srplusgwr = srplusgwr.dropna()

def get_30min_array(staticdata, dconfig, s, nan=-99):
    """
    Create a numpy array in resolution of 30min(720 x 360 of WaterGAP data) from a pd.Series

    Parameters
    ----------
    s : pd.Series or 'flowdir'
        pandas Series is mapped via index (arcid) or via 'flowdir' via inputdir
    nan : int
        value which represents nan

    Returns
    -------
    np.array
    """
    array = np.full((360, 720), nan)
    wg_input = staticdata['wg_input']
    aoi = dconfig.aoi
    if isinstance(s, pd.Series):
        s.name = 'variable'
        df = wg_input.merge(s, left_index=True, right_index=True)  # Append basic information
        flowdir = False
    elif s == 'flowdir':
        df = wg_input.rename(columns={"G_FLOWDIR.UNF2": "variable"})
        flowdir = True
    else:
        raise Exception('not implemented')

    # Convert df to numpy array
    for x in df.itertuples():
        array[x._2 - 1, x._3 - 1] = x.variable

    # Subset array to intersect with area of interest
    ar = array[int(360 - (aoi[1][1] + 90) * 2):int(360 - (aoi[1][0] + 90) * 2),
         int((aoi[0][0] + 180) * 2): int((aoi[0][1] + 180) * 2)]
    if flowdir:
        # avoid flow out of aoi
        # top border
        ar[0, :][ar[0, :] >= 32] = 0
        # left border
        ar[:, 0][ar[:, 0] == 32] = 0
        ar[:, 0][ar[:, 0] == 16] = 0
        ar[:, 0][ar[:, 0] == 8] = 0
        # right border
        ar[:, -1][ar[:, -1] == 128] = 0
        ar[:, -1][ar[:, -1] == 1] = 0
        ar[:, -1][ar[:, -1] == 2] = 0
        # bottom border
        ar[-1, :][ar[-1, :] == 8] = 0
        ar[-1, :][ar[-1, :] == 4] = 0
        ar[-1, :][ar[-1, :] == 2] = 0
    return ar

reliable_surfacerunoff_ar = get_30min_array(staticdata=staticdata, dconfig=dconfig,
                                            s=srplusgwr , nan=np.nan)

#Use DownScaleArray.py code
# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(reliable_surfacerunoff_ar,
#                                                     status='30min_srplusgwr_eu_200308'
#                                                     )

#4.d.--------- Compute initial 15-sec runoff-based cell discharge ######################################################
#runoffbased_celldis_15s_ar = self.get_runoff_based_celldis(reliable_surfacerunoff_ar, month=month, yr=year)
kwargs['month'] = month
kwargs['year'] = year

#d.ii. inverse distance interpolation from lr to intermediate resolution of 0.1 degree---------------------------------
#create_inmemory_30min_pointds---------------
inp=reliable_surfacerunoff_ar
all=True
aoi = dconfig.aoi
coords = staticdata['coords']

df = None
inptype = 'other'

drv = gdal.GetDriverByName('Memory')
ds = drv.Create('runofftemp', 0, 0, 0, gdal.GDT_Unknown)
lyr = ds.CreateLayer('runofftemp', None, ogr.wkbPoint)
field_defn = ogr.FieldDefn('variable', ogr.OFTReal)
lyr.CreateField(field_defn)

#if 'all' in kwargs:
for idx, value in np.ndenumerate(inp):
    x = aoi[0][0] + (idx[1] / 2) + 0.25  # Create point in the middle of cells (0.25 arc-degs from the edge)
    y = aoi[1][1] - ((idx[0] / 2) + 0.25)
    if not np.isnan(value):
        feat = ogr.Feature(lyr.GetLayerDefn())
        # irow, icol = self.wg_input.data.loc[x.Index, ['GR.UNF2', 'GC.UNF2']]
        feat.SetField("variable", value)
        pt = ogr.Geometry(ogr.wkbPoint)
        pt.SetPoint(0, x, y)
        feat.SetGeometry(pt)
        lyr.CreateFeature(feat)
        feat.Destroy()
tmp_ds = ds

#d.iii. tmp_interp = interpolation_to_grid(tmp_ds, '6min')--------------------------------------------------------------
resolution = '6min'
if resolution == '6min':
    width = (aoi[0][1] - aoi[0][0]) * 10
    height = (aoi[1][1] - aoi[1][0]) * 10

outputbounds = [aoi[0][0], aoi[1][1], aoi[0][1], aoi[1][0]]
alg = "invdistnn:power=2.0:smoothing=0.0:radius=1.8:max_points=9:nodata=-99"
if 'alg' in kwargs:
    alg = kwargs.pop('alg')
out_raster_srs = osr.SpatialReference()
out_raster_srs.ImportFromEPSG(4326)
# Perform interpolation
go = gdal.GridOptions(format='MEM',
                      outputType=gdal.GDT_Float32,
                      layers='runofftemp',
                      zfield='variable',
                      outputSRS=out_raster_srs,
                      algorithm=alg,
                      width=width,
                      height=height,
                      outputBounds=outputbounds)
# Create output grid
outr = gdal.Grid('outr', ds, options=go)
res = outr.ReadAsArray().copy()
# del outr
#
# del tmp_ds
# del ds
#tmp_interp=res
res[res == -99] = np.nan

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(res,
#                                                     status='6min_srplusgwr_eu_200308'
#                                                     )
tmp_smooth = res

#d.iv. Disaggregate from 6 min to 15 s----------------------------------------------------------------------------------
#interpolated_smooth_15s = self.disaggregate(tmp_smooth, 24)
a = np.repeat(tmp_smooth.data, 24, axis=0)
interpolated_smooth_15s = np.repeat(a, 24, axis=1)
# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(interpolated_smooth_15s,
#                                                     status='15s_srplusgwr_eu_200308'
#                                                     )
# d.v. Remove 15-sec cells where original HydroSHEDS pixel area raster is NoData------------------------------------------
#masked_diss_tmp_smooth = mask_wg_with_hydrosheds(interpolated_smooth_15s)
hydrosheds_ar = staticdata['pixelarea']
array = np.full(hydrosheds_ar.shape, np.nan)
hydrosheds_geotrans = staticdata['hydrosheds_geotrans']
coloffset = int(round(dconfig.aoi[0][0] - hydrosheds_geotrans[0]) // 0.5 * 120 * -1)
rowoffset = int(round(dconfig.aoi[1][1] - hydrosheds_geotrans[3]) // 0.5 * 120)
offset = interpolated_smooth_15s[rowoffset:, coloffset:]
rowix = array.shape[0] - offset.shape[0]
colix = array.shape[1] - offset.shape[1]
if rowix == 0:
    rowix = array.shape[0]
if colix == 0:
    colix = array.shape[1]
wgdata = offset[:rowix, :colix]
array[~np.isnan(hydrosheds_ar)] = wgdata[~np.isnan(hydrosheds_ar)]
masked_diss_tmp_smooth = array

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(masked_diss_tmp_smooth,
#                                                     status='15s_srplusgwr_eu_200308_masked'
#                                                     )

# d.vi. Compute runoff-based discharge in the cell (i.e. convert runoff from mm to m3/s)--------------------------------
#conv = self.convert_runoff_to_dis(masked_diss_tmp_smooth, **kwargs)
if dconfig.mode == 'ts':
    division_term = (daysinmonth_dict[kwargs['month']] * 24 * 60 * 60)
area = staticdata['pixelarea']
data = (masked_diss_tmp_smooth * area * 1000 / division_term)
conv=data

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(conv,
#                                                     status='15s_srplusgwr_eu_200308_masked_m3s'
#                                                     )

# d.vii. Correct runoff-based discharge for changes in storage of global lakes and reservoirs---------------------------
# (i.e., redistribute volume changes from global water bodies' pour points to all cells intersecting with
# that global lake)
if dconfig.correct_global_lakes:
    globallakes_fraction_15s = staticdata['globallakes_fraction_15s']

    globallakes_addition_15s_ar = get_30min_array(staticdata=staticdata,
                                                  dconfig=dconfig,
                                                  s=taskdata_dict['2003_8']['globallakes_addition'],
                                                  nan=float(0))

    # DownScaleArray(dconfig,
    #                dconfig.aoi,
    #                write_raster_trigger=True).load_data(globallakes_addition_15s_ar,
    #                                                     status='30min_globallakes_addition_eu_200308_masked_km3mo')


    # Disaggregate global lakes redistribution values from 30 min pd.Series in km3/mo to 15 arc arrays in m3/s
    lake_redis_m3s = get_30min_array(staticdata=staticdata,
                                     dconfig=dconfig,
                                     s=(taskdata_dict['2003_8']['globallakes_addition']/(daysinmonth_dict[kwargs['month']] * 24 * 60 * 60)) * 1000000000,
                                     nan=float(0))

    def disaggregate_ar(what, factor):
        """ Takes a 2D numpy array and repeat the value of one pixel by factor along both axis

        :param what: 2D Array which should be disaggregated
        :type what: np.ndarray
        :param factor: how often the value is repeated
        :type factor: int
        :return: disaggregated 2D array which values are repeated by the factor
        :rtype: np.ndarray
        """
        a = np.repeat(what, factor, axis=0)
        b = np.repeat(a, factor, axis=1)
        return b

    globallakes_addition_ar_15s_m3s = disaggregate_ar(lake_redis_m3s,factor=120)

    # DownScaleArray(dconfig,
    #                dconfig.aoi,
    #                write_raster_trigger=True).load_data(globallakes_addition_ar_15s_m3s,
    #                                                     status='15s_globallakes_addition_eu_200308_masked_m3s')

    # Redistribute storage change based on fraction of lakes in LR cell intersecting with HR cell
    conv = conv - (globallakes_fraction_15s * globallakes_addition_ar_15s_m3s)

    # Convert negative runoff-based discharge values to 0
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        conv = np.where(conv < 0, 0, conv)

    runoffbased_celldis_15s_ar = conv

    # DownScaleArray(dconfig,
    #                dconfig.aoi,
    #                write_raster_trigger=True).load_data(runoffbased_celldis_15s_ar,
    #                                                     status='15s_srplusgwr_eu_200308_masked_m3s_lkrescor')

# 4.e. Compute initial correction values in each 30-min cell ##########################################################################
netdis_30min_series = taskdata_dict['2003_8']['netdis_30min_series']
netdis_30min_series.name = 'variable'

# correction_grid_30min = self.calculate_lr_correctionvalues(runoffbased_celldis_15s_ar=runoffbased_celldis_15s_ar,
#                                                            netdis_30min_series=netdis_30min_series,
#                                                            month=month, yr=year)
# e.i. aggregate-sum 15-sec runoff-based cell discharge to 30-min runoff-based cell discharge (m3/s)--------------------
def aggsum(what, factor, zeroremove=True):
    what = what.copy()
    what[np.isnan(what)] = 0
    result = np.einsum('ijkl->ik',  #Sum over the j and l axes, resulting in a 2D array with shape (i, k).
                       what.reshape(what.shape[0] // factor, factor, -1, factor)) #Reshape from (17280, 22800) to (144, 120, 190, 120) (ijkl)
    if zeroremove:
        result[result == 0] = np.nan
    return result

runoffbased_celldis_30min_ar = aggsum(what=runoffbased_celldis_15s_ar, factor=120, zeroremove=False)

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(runoffbased_celldis_30min_ar,
#                                                     status='30min_srplusgwr_eu_200308_masked_m3s_lkrescor_reaggregated')

#e.ii. Convert 30-min net runoff to m3/s (from km3/month if self.dconfig.mode == 'ts', else from km3/yr)----------------
if dconfig.mode == 'ts':
    netdis_30min_m3s_ar = ((get_30min_array(staticdata=staticdata,dconfig=dconfig,
                                            s=netdis_30min_series, nan=np.nan)/
                            (daysinmonth_dict[kwargs['month']] * 24 * 60 * 60)) * 1000000000)

    # DownScaleArray(dconfig,
    #                dconfig.aoi,
    #                write_raster_trigger=True).load_data(netdis_30min_m3s_ar,
    #                                                     status='30min_netdis_eu_200308_m3s')

#e.iii. Control for proportion of actual land (vs sea) in 30-min cell---------------------------------------------------
landratio_corr = staticdata['landratio_corr']
netdis_30min_m3s_ar *= landratio_corr

#e.iv. Fill NAs in net runoff array with values from aggregated runoff-based cell discharge-----------------------------
def stack(upper, lower):
    """
    Fill nan values in upper with values of lower

    Parameters
    ----------
    upper : np.array
    lower : np.array

    Returns
    -------
    np.array
    """
    tobefilled = np.isnan(upper)
    new = upper
    new[tobefilled] = lower[tobefilled]
    return new

netdis_30min_m3s_ar = stack(upper=netdis_30min_m3s_ar, lower=runoffbased_celldis_30min_ar)

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(netdis_30min_m3s_ar,
#                                                     status='30min_netdis_eu_200308_m3s_filled')

#e.v. Compute correction value stemming from each cell (to be routed downstream)----------------------------------------
dif_dis_30min = netdis_30min_m3s_ar - runoffbased_celldis_30min_ar
correction_grid_30min = dif_dis_30min

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(dif_dis_30min,
#                                                     status='30min_difdis_eu_200308_m3s')

#f. Compute what proportion of the 30-min correction value should apply to each 15-sec cell############################################
#correction_weights_15s = self.get_corrweight(runoffbased_celldis_15s_ar)

#f.i. Compute actual flow-accumulated discharge/Flow-accumulate HR runoff-based cell discharge.-------------------------
def flow_acc(staticdata, dis15s):
    # Get/run flow accumulation
    newar = staticdata['flowacc'].get(dis15s)
    newar[np.isnan(staticdata['pixelarea'])] = np.nan
    return newar

runoffbased_dis_15s = flow_acc(staticdata=staticdata, dis15s=runoffbased_celldis_15s_ar)

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(runoffbased_dis_15s,
#                                                     status='15s_srplusgwr_eu_200308_masked_m3s_lkrescor_acc')

#f.ii. Compute sum of discharge from all HR cells in each LR cell-------------------------------------------------------
dis_fg_sum_30min = aggsum(runoffbased_dis_15s, 120, zeroremove=False)

# Replace 0s with 1 to avoid dividing by 0, but won't influence outcome (dividing 0 by 1)
dis_fg_sum_30min[dis_fg_sum_30min == 0] = 1

#f.iii. Compute proportion of discharge from LR cell contained in each HR cell------------------------------------------
dis_corr_weight = runoffbased_dis_15s / disaggregate_ar(dis_fg_sum_30min, 120)
correction_weights_15s = dis_corr_weight

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(dis_corr_weight,
#                                                     status='15s_discorrweight_eu_200308')

#g. Change correction values in 30-min cells to apply an even greater proportion of the correction on large rivers##########################
# (if large_river_corr is True)
#g.i. Apply correct_dis()-------------------------------------------------------------------------------------------
# corrected_dis_15s = self.correct_dis(correction_grid_30min=correction_grid_30min,
#                                      correction_weights_15s=correction_weights_15s,
#                                      runoffbased_celldis_15s_ar=runoffbased_celldis_15s_ar,
#                                      month=month)
# Apply correction weights to correction grid
celldis_correctionvalue_15s = disaggregate_ar(correction_grid_30min.data, 120) * correction_weights_15s.data

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(celldis_correctionvalue_15s,
#                                                     status='15s_celldis_correctionvalue_raw_eu_200308')

if dconfig.correction_threshold is not None:
    ctar = staticdata['upstream_pixelarea'] * dconfig.correction_threshold
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # Create a grid to know which cells have negative correction
        corr_sign = np.where(celldis_correctionvalue_15s < 0, -1, 1)
    # Make sure the absolute value of the correction weights remain under the correction_threshold
    celldis_correctionvalue_15s = np.minimum(np.abs(celldis_correctionvalue_15s), ctar)
    # Re-assign correct sign to each correction weight
    celldis_correctionvalue_15s = celldis_correctionvalue_15s * corr_sign
    del corr_sign

    # DownScaleArray(dconfig,
    #                dconfig.aoi,
    #                write_raster_trigger=True).load_data(ctar,
    #                                                     status='15s_correction_threshold_eu_200308')
    # DownScaleArray(dconfig,
    #                dconfig.aoi,
    #                write_raster_trigger=True).load_data(celldis_correctionvalue_15s,
    #                                                     status='15s_celldis_correctionvalue_capped_eu_200308')

# Apply correction values to flow-accumulated runoff-based discharge
corrected_celldis = runoffbased_celldis_15s_ar + celldis_correctionvalue_15s

# Accumulate corrected cell discharge to yield actual dischare
corrected_dis = flow_acc(staticdata=staticdata, dis15s=corrected_celldis)

# manual correction of negative discharge
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    corrected_dis[corrected_dis < 0] = 0

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(corrected_dis,
#                                                     status='15s_corrected_dis_ini_eu_200308')
corrected_dis_15s = corrected_dis

#g.ii.	Apply get_largerivers_correction_grid():-----------------------------------------------------------------------
wg_dis_30min = taskdata_dict['2003_8']['dis']
wg_dis_30min.name = 'variable'
precorrected_dis = corrected_dis_15s
# correction_grid_30min = self.get_largerivers_correction_grid(
#     wgdis_series_30min=wg_dis_30min,
#     correction_grid_30min=correction_grid_30min,
#     precorrected_dis=corrected_dis_15s,
#     month=month)
# del corrected_dis_15s

flowdir = get_30min_array(staticdata=staticdata,dconfig=dconfig, s='flowdir')
largerivers_mask_30min = staticdata['largerivers_mask']

# Convert discharge to m3/s (from km3/month if mode==ts or km3/yr if mode==long-term average
if dconfig.mode == 'ts':
    wgdis_30min_m3s = (get_30min_array(staticdata=staticdata,dconfig=dconfig, s=wg_dis_30min, nan=np.nan)/
                       (daysinmonth_dict[kwargs['month']] * 24 * 60* 60)) * 1000000000
# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(wgdis_30min_m3s,
#                                                     status='30min_WGdis_eu_200308_m3s')

# Compute net discharge in large river 30 min cells from WG discharge data
wgdis_largerivers_30min = wgdis_30min_m3s * largerivers_mask_30min
net_wgdis_largerivers_30min = (wgdis_largerivers_30min
                               - get_inflow_sum(in_valuegrid=wgdis_largerivers_30min,
                                                in_flowdir=flowdir)
                               )

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(net_wgdis_largerivers_30min,
#                                                     status='30min_net_WGdis_largerivers_eu_200308_m3s')

# Compute maximum 15 sec accumulated pre-corrected discharge in each large river 30 min cell
cell_pourpixel = staticdata['cell_pourpixel']
precorrected_dis = precorrected_dis * cell_pourpixel

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(precorrected_dis,
#                                                     status='15s_precorrected_dis_prpix30min_eu_200308_m3s')

def aggmax(what, factor):
    """

    :param what:
    :type what:
    :param factor:
    :type factor:
    :return:
    :rtype:
    """
    b = np.split(what, what.shape[1] / factor, axis=1)
    c = [x.reshape((-1, factor * factor)) for x in b]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        d = [np.nanmax(x, axis=1) for x in c]
    return np.column_stack(d)

accumulated_precorrecteddis_30min = aggmax(precorrected_dis, 120)
accumulated_precorrecteddis_30min_largerivers = accumulated_precorrecteddis_30min * largerivers_mask_30min

# Compute net pre-corrected discharge in each large river 30 min cell
net_precorrecteddis_largerivers_30min = (accumulated_precorrecteddis_30min_largerivers
                                         - get_inflow_sum(in_valuegrid=accumulated_precorrecteddis_30min_largerivers,
                                                          in_flowdir=flowdir)
                                         )

# Compute difference in net discharge from WG and from the accumulated corrected discharge
net_dis_dif_largerivers_30min = net_wgdis_largerivers_30min - net_precorrecteddis_largerivers_30min

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(net_dis_dif_largerivers_30min,
#                                                     status='30min_net_dis_dif_largerivers_eu_200308_m3s')

# Transfer net_dis_dif from outside the large-rivers mask to large rivers downstream
gapmask_30min = 1 - largerivers_mask_30min
transfer_value_grid = gapmask_30min * net_dis_dif_largerivers_30min #This is equal to 0, aside where there are issues with the large river mask. This should be conducted at 15 arc-sec?
# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(transfer_value_grid,
#                                                     status='30min_transfer_value_largerivers_eu_200308_m3s')

gap_flowdir_30min = flowdir * (1 - largerivers_mask_30min)
gap_flowdir_30min[flowdir == -99] = -99

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(gap_flowdir_30min,
#                                                     status='30min_flowdir_outsidelargerivers_eu')

gap_flowacc = staticdata['30mingap_flowacc']
transfer_accu_grid = FlowAccTT(in_flowdir=gap_flowdir_30min,
                               in_static_flowacc=gap_flowacc,
                               pad=True).get(
    in_valuegrid=transfer_value_grid,
    no_negative_accumulation=False)

# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(transfer_accu_grid,
#                                                     status='30min_transfer_grid_accu_eu')


# Compute final net discharge correction value for large rivers
new_diff_dis_30min = correction_grid_30min + ((net_dis_dif_largerivers_30min + transfer_accu_grid)
                                              * largerivers_mask_30min)
correction_grid_30min = new_diff_dis_30min
# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(new_diff_dis_30min,
#                                                     status='30min_newdifdis_eu_200308_m3s')

#h. Shift correction values downstream at 30-min#######################################################################
# if self.dconfig.corr_grid_shift:
#     correction_grid_30min = self.shift_correction_grid(correction_grid_30min)
# For each cell, partially shift the correction value to the next LR downstream cell
# if the maximum HR upstream area in the next LR downstream cell is at least 0.9*maximum HR upstream area in the cell
# This process is to focus correction even more strongly on large rivers
corr_grid = ((correction_grid_30min * staticdata['keepgrid'])
             + get_inflow_sum(in_valuegrid=(correction_grid_30min * staticdata['shiftgrid']),
                              in_flowdir=flowdir)
             )
# DownScaleArray(dconfig,
#                dconfig.aoi,
#                write_raster_trigger=True).load_data(corr_grid,
#                                                     status='30min_newdifdis_eu_200308_m3s_shifted')
correction_grid_30min = corr_grid

#j. Apply correction at 15-sec and accumulate corrected net discharge downstream########################################
# corrected_dis_15s = self.correct_dis(correction_grid_30min=correction_grid_30min,
#                                      correction_weights_15s=correction_weights_15s,
#                                      runoffbased_celldis_15s_ar=runoffbased_celldis_15s_ar,
#                                      correction_threshold=self.dconfig.correction_threshold,
#                                      month=month)
# Apply correction weights to correction grid
###########################\
############################## NEED TO CHANGE NAs in correction grid to 0s before running correct_dist again

celldis_correctionvalue_15s = disaggregate_ar(correction_grid_30min.data, 120) * correction_weights_15s.data

if dconfig.correction_threshold is not None:
    ctar = staticdata['upstream_pixelarea'] * dconfig.correction_threshold
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        # Create a grid to know which cells have negative correction
        corr_sign = np.where(celldis_correctionvalue_15s < 0, -1, 1)
    # Make sure the absolute value of the correction weights remain under the correction_threshold
    celldis_correctionvalue_15s = np.minimum(np.abs(celldis_correctionvalue_15s), ctar)
    # Re-assign correct sign to each correction weight
    celldis_correctionvalue_15s = celldis_correctionvalue_15s * corr_sign
    del corr_sign

# Apply correction values to flow-accumulated runoff-based discharge
corrected_celldis = runoffbased_celldis_15s_ar + celldis_correctionvalue_15s

# Accumulate corrected cell discharge to yield actual dischare
corrected_dis = flow_acc(staticdata=staticdata, dis15s=corrected_celldis)

# manual correction of negative discharge
with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    corrected_dis[corrected_dis < 0] = 0

DownScaleArray(dconfig,
               dconfig.aoi,
               write_raster_trigger=True).load_data(corrected_dis,
                                                    status='15s_corrected_dis_final_eu_200308')