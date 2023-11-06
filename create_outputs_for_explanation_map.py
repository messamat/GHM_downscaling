#### Functions to save maps #############################################################################################
from open.WGReadingFunctions import read_variable, InputDir

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

def export_wgseries_to_tif(staticdata, dconfig, s, out_dir, out_filename):
    reliable_surfacerunoff_ar = get_30min_array(staticdata=staticdata, dconfig=dconfig,
                                                s=s, nan=np.nan)

    dtype = gdal.GDT_Float32

    if 'dtype' in kwargs:
        dtype = kwargs['dtype']

    write_raster_specs = {
                '30min': 2,
                '6min': 10,
                '30sec': 120,
                '15s': 240
            }

    rmulti = write_raster_specs['30min']
    # Get number of cols and rows based on extent and conversion factor
    no_cols = (aoi[0][1] - aoi[0][0]) * rmulti
    no_rows = (aoi[1][1] - aoi[1][0]) * rmulti
    cellsize = 1 / rmulti
    leftdown = (aoi[0][0], aoi[1][0])  # lower-left corner of extent
    grid_specs = (int(no_rows), int(no_cols), cellsize, leftdown)

    rows, cols, size, origin = grid_specs
    originx, originy = origin

    driver = gdal.GetDriverByName('GTiff')
    name = os.path.join(out_dir, out_filename)
    out_raster = driver.Create(name, cols, rows, 1, dtype)
    out_raster.SetGeoTransform((originx, size, 0, originy, 0, size))
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(reliable_surfacerunoff_ar[::-1])
    out_band.SetNoDataValue(-99.)
    out_raster_srs = osr.SpatialReference()
    out_raster_srs.ImportFromEPSG(4326)
    out_raster.SetProjection(out_raster_srs.ExportToWkt())

    out_band.FlushCache()
    out_raster.FlushCache()
    outband = None
    out_raster = None

#RunDownscaling.py #####################################################################################################

import os
from inspect import getsourcefile
from osgeo import gdal, osr
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

explanation_outdir = os.path.join(rootdir, 'results', 'algorithm_explanation')
if not os.path.exists(explanation_outdir ):
    os.mkdir(explanation_outdir)

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
                            write_raster=False,
                            write_result='nc',
                            mode='ts',
                            continent=continent,
                            constants_folder=constants_folder,
                            pois=pois,
                            runoff_src='srplusgwr',
                            correct_global_lakes=True,
                            sr_interp_wg_nas = False,
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

temp_downscalearray = DownScaleArray(config, dconfig.aoi, **kwargs) #Create standardized placeholder for data
surfacerunoff_based_dis = None
corrected_dis = None
correction_weights_15s = None
correction_grid_30min = None

#--------------------------- Run down.prepare() ---------------------------------------------------------------------
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
data = {}
timestep=1
for yr in range(dconfig.startyear, dconfig.endyear + 1):
        for mon in range(1, 13):
            print(timestep)
            data["{0}_{1}".format(yr, mon)] = {
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
                data["{0}_{1}".format(yr, mon)]['globallakes_addition'] = (
                    wg.globallakes_addition.loc)[slice(None), yr, mon]
            timestep += 1

# Write list of tasks to pickle
for t in data:
    out_pickle = os.path.join(dconfig.temp_dir,
                              'data_task{:03d}.pickle'.format(data[t]['i'])
                              )
    with open(out_pickle, 'wb') as f:
        pickle.dump(data[t], f)

if config:
    dconfig.pickle()

del hydrosheds
del wg


#Compute total runoff
srplusgwr = data['2003_8']['sr'] + data['2003_8']['gwrunoff']
srplusgwr = srplusgwr.dropna()


#Save layers ###################################################################################################
######FROM WGData #############################################################################################
read_variable_kwargs = {
    "wg_simoutput_path": config.wg_out_path,
    "timestep": 'month',
    "startyear": config.startyear,
    "endyear": config.endyear
}

if not os.path.exists(os.path.join(explanation_outdir, 'WG_input_G_GLOLAK.tif')):
    GLOLAK_WET = InputDir().read(in_path=config.wg_in_path,
                             arcid_folderpath=config.constants_folder,
                             explain=False,
                             files=['G_GLOLAK.UNF0', 'G_GLOWET.UNF0','GR.UNF2'],
                             additional_files=False, cellarea=True).data

    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=GLOLAK_WET.loc[:, 'G_GLOLAK.UNF0'],
                           out_dir=explanation_outdir,
                           out_filename='WG_input_G_GLOLAK.tif')

    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=GLOLAK_WET.loc[:, 'G_GLOWET.UNF0'],
                           out_dir=explanation_outdir,
                           out_filename='WG_input_G_GLOWET.tif')

if not os.path.exists(os.path.join(explanation_outdir, 'WG_input_GCONTFREQ_200308.tif')): #Proportion of pixel in continental area
    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=staticdata['wg_input'].loc[:, 'GCONTFREQ.UNF0'],
                           out_dir=explanation_outdir,
                           out_filename='WG_input_GCONTFREQ_200308.tif')

if not os.path.exists(os.path.join(explanation_outdir, 'WG_input_G_SURFACE_RUNOFF_200308.tif')):  #sr - surface_runoff - 'G_SURFACE_RUNOFF_'
    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=data['2003_8']['sr'],
                           out_dir=explanation_outdir,
                           out_filename='WG_input_G_SURFACE_RUNOFF_200308.tif')

if not os.path.exists(os.path.join(explanation_outdir, 'WG_input_G_GW_RUNOFF_mm_200308.tif')): #gwrunoff - gw_runoff - 'G_GW_RUNOFF_mm_'
    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=data['2003_8']['gwrunoff'],
                           out_dir=explanation_outdir,
                           out_filename='WG_input_G_GW_RUNOFF_mm_200308.tif')

if not os.path.exists(os.path.join(explanation_outdir,
                                   'WG_input_G_RUNOFF_200308.tif')):  # netdis_30min_series (DryverDownscalingWrapper) - total_runoff (wg object) -'G_RUNOFF_' (wg file name)
    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=data['2003_8']['netdis_30min_series'],
                           out_dir=explanation_outdir,
                           out_filename='WG_input_G_RUNOFF_200308.tif')

if not os.path.exists(os.path.join(explanation_outdir,
                                   'WG_input_G_RIVER_AVAIL_200308.tif')):  # dis - dis-'G_RIVER_AVAIL_'
    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=data['2003_8']['dis'],
                           out_dir=explanation_outdir,
                           out_filename='WG_input_G_RIVER_AVAIL_200308.tif')

if not os.path.exists(os.path.join(explanation_outdir,
                                   'WG_input_G_CELL_RUNOFF_200308.tif')):  # dis - dis-'G_RIVER_AVAIL_'
    cell_runoff = read_variable(var='G_CELL_RUNOFF_', arcid_folderpath=config.constants_folder,
                                **read_variable_kwargs).data.set_index(['arcid', 'year', 'month'])
    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=cell_runoff.loc[:, 2003, 8]['net_cell_runoff'],
                           out_dir=explanation_outdir,
                           out_filename='WG_input_G_CELL_RUNOFF_200308.tif')

if not os.path.exists(os.path.join(explanation_outdir,
                                   'G_GLO_LAKE_STORAGE_km3_200308')):  # dis - dis-'G_RIVER_AVAIL_'
    glolak = read_variable(var='G_GLO_LAKE_STORAGE_km3_', arcid_folderpath=config.constants_folder,
                           **read_variable_kwargs).data
    diff_glolak = (glolak.set_index(['year', 'month', 'arcid']).unstack() -
                   glolak.set_index(['year', 'month', 'arcid']).unstack().shift(1)).stack()

    glores = read_variable(var='G_RES_STORAGE_km3_', arcid_folderpath=config.constants_folder,
                           **read_variable_kwargs).data
    diff_glores = (glores.set_index(['year', 'month', 'arcid']).unstack() -
                   glores.set_index(['year', 'month', 'arcid']).unstack().shift(1)).stack()

    diff_gloresglolak = ((diff_glores + diff_glolak).reset_index().
                         set_index(['arcid', 'year', 'month']).sort_index())

    for mon in [7,8]:
        export_wgseries_to_tif(staticdata=staticdata,
                               dconfig=dconfig,
                               s=glolak.set_index(['arcid','year', 'month']).loc[:, 2003, mon]['variable'],
                               out_dir=explanation_outdir,
                               out_filename='G_GLO_LAKE_STORAGE_km3_20030{}.tif'.format(mon))
        export_wgseries_to_tif(staticdata=staticdata,
                               dconfig=dconfig,
                               s=glores.set_index(['arcid','year', 'month']).loc[:, 2003, mon]['variable'],
                               out_dir=explanation_outdir,
                               out_filename='G_GLO_RES_STORAGE_km3_20030{}.tif'.format(mon))
    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=diff_gloresglolak.loc[:, 2003, 8]['variable'],
                           out_dir=explanation_outdir,
                           out_filename='WGData_intermediate_diff_gloresglolak_200308.tif'.format(mon))

#### ANALYZE LAKE CORRECTION IN CELL RUNOFF
# Subset analysis extent to "area_of_interest" ------------------------------------------------------------------
if not os.path.exists('WGData_intermediate_cell_runoff_afterglolakresredis_2003008.tif'):
    cell_runoff = read_variable(var='G_CELL_RUNOFF_', arcid_folderpath=config.constants_folder,
                                **read_variable_kwargs).data.set_index(['arcid', 'year', 'month'])
    # cell runoff with extra treatment for global lakes and reservoirs
    modrkwargs = read_variable_kwargs.copy()
    modrkwargs['startyear'] = config.startyear - 1

    # Compute change in reservoir or lake storage at each time step compared to previous time step
    glolak = read_variable(var='G_GLO_LAKE_STORAGE_km3_', arcid_folderpath=config.constants_folder,
                           **modrkwargs).data
    diff_glolak = (glolak.set_index(['year', 'month', 'arcid']).unstack() -
                   glolak.set_index(['year', 'month', 'arcid']).unstack().shift(1)).stack()

    glores = read_variable(var='G_RES_STORAGE_km3_', arcid_folderpath=config.constants_folder,
                           **modrkwargs).data
    diff_glores = (glores.set_index(['year', 'month', 'arcid']).unstack() -
                   glores.set_index(['year', 'month', 'arcid']).unstack().shift(1)).stack()

    diff_gloresglolak = ((diff_glores + diff_glolak).reset_index().
                         set_index(['arcid', 'year', 'month']).sort_index())

    if "area_of_interest" in kwargs:
        aoi = kwargs['area_of_interest']
        arcids = pd.read_csv(os.path.join(config.constants_folder, 'xyarcid.csv')).set_index(["X", "Y"])
        relative_arcids_df = arcids.sort_index().loc[(slice(aoi[0][0], aoi[0][1]),
                                                      slice(aoi[1][0], aoi[1][1])), 'arcid']
        relative_arcids = relative_arcids_df.values.tolist()
        read_variable_kwargs['arcid_list'] = relative_arcids
        coords = relative_arcids_df.reset_index().set_index('arcid')

    # Read global lakes and reservoirs redistribution table and subset it if a set of cell IDs were provided
    # to subset WG data
    if 'arcid_list' in read_variable_kwargs:
        redist_glwd = pd.read_csv(os.path.join(config.constants_folder,
                                               'glolakandresredistribution.csv'))
        redist_glwd = redist_glwd.loc[[x in read_variable_kwargs['arcid_list'] for x in redist_glwd.arcid], :]
    else:
        redist_glwd = pd.read_csv(os.path.join(config.constants_folder,
                                               'glolakandresredistribution.csv'))

    # Prepare time series grids to redistribute changes in lake or reservoir storage at every time step
    # from the outflow cell to all cells intersecting with each global lake and reservoir
    globallakes_additionlist = []
    for glwdunit in redist_glwd.glwdunit.unique():  # For each global lake
        redistmp = redist_glwd[redist_glwd.glwdunit == glwdunit].set_index('arcid')
        outflowcell = redistmp.outflowcell.iloc[0]
        # Skip global lake if outflow cell (pourpoint) lies outside of area of interest
        if outflowcell not in read_variable_kwargs['arcid_list']:
            continue
        # Extract difference in lake or reservoir volume at outflow cell
        globallakes_addition = diff_gloresglolak.loc[(outflowcell, slice(None), slice(None)), :].reset_index()
        globallakes_addition = globallakes_addition.loc[
            globallakes_addition.year >= config.startyear, 'variable'].values  # Subset years within period of interest

        for arcid in redistmp.index.get_level_values('arcid'):  # For each cell intersecting with the global lake
            frac = redistmp.loc[arcid, 'fracarea']  # Extract the fraction of that cell that intersects with the lake
            # Remove storage change from outflow cell and redistribute to all cells that intersect with lake
            # (NEED to make sure this isn't redundant with step in DryverDownscaling and that the sign is correct)
            if arcid == outflowcell:
                cell_runoff.loc[(arcid, slice(None), slice(None)), 'net_cell_runoff'] = (
                        cell_runoff.loc[(arcid, slice(None), slice(None)), 'net_cell_runoff'] + globallakes_addition)

            cell_runoff.loc[(arcid, slice(None), slice(None)), 'net_cell_runoff'] = (
                    cell_runoff.loc[(arcid, slice(None), slice(None)), 'net_cell_runoff'] - (globallakes_addition * frac))

            # Create a df to record storage change for each cell and time step
            years = [x for x in range(config.startyear, config.endyear + 1)]
            months = [x for x in range(1, 13)]
            mix = pd.MultiIndex.from_product([[int(arcid)], years, months],
                                             names=['arcid', 'years', 'month'])
            globallakes_additionlist.append(pd.Series(globallakes_addition * frac, mix))

    cell_runoff = cell_runoff.reset_index()
    globallakes_addition = pd.concat(globallakes_additionlist)

    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=cell_runoff.set_index(['arcid', 'year', 'month']).loc[:, 2003, 8]['net_cell_runoff'],
                           out_dir=explanation_outdir,
                           out_filename='WGData_intermediate_cell_runoff_afterglolakresredis_2003008.tif')

    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=globallakes_addition.loc[:, 2003, 8],
                           out_dir=explanation_outdir,
                           out_filename='WGData_intermediate_globallakes_addition_2003008.tif')


######FROM WGData #############################################################################################
if not os.path.exists(os.path.join(explanation_outdir, 'srplusgwr_eu_30min_200308.tif')):
    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=srplusgwr,
                           out_dir=explanation_outdir,
                           out_filename='srplusgwr_eu_30min_200308.tif')

if not os.path.exists(os.path.join(explanation_outdir, 'mean_wgland_fraction_30min.tif')):
    export_wgseries_to_tif(staticdata=staticdata,
                           dconfig=dconfig,
                           s=staticdata['mean_land_fraction'],
                           out_dir=explanation_outdir,
                           out_filename='mean_wgland_fraction_30min.tif')








