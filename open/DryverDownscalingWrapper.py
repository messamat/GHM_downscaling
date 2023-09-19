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


class DryverDownscalingWrapper:
    """
    Dryver Downscaling object takes a configuration, prepares the downscaling process and is able to start it

    Parameters
    ----------
    config : open.DryverDownscalingConfig.DownscalingConfig
        Configuration object to run downscaling.
    **kwargs : dict, optional
        Keyword arguments

    Attributes
    ----------
    mode : {'longterm_avg', 'ts'}
        mode of downscaling either longterm_avg = long term average or ts = timeseries
    dconfig : open.DryverDownscalingConfig.DownscalingConfig
        Configuration object to run downscaling
    kwargs : dict
        kwargs updated with kwargs from DownscalingConfig
    wg : WGData.WGData
        data from WaterGAP
    hydrosheds : HydroSHEDSData.HydroSHEDSData
        data from HydroSHEDS
    daysinmonth : dict
        Dict with key = month as numeric and value = number of days
    tempdsarray : DownScaleArray
        Container for temporary results is able to write raster and pickle the data
    surfacerunoffbaseddis : None or DownScaleArray
        Initialized with None. Holds in downscaling the surface runoff based discharge
    correcteddis : None or DownScaleArray
        Initialized with None. Holds in downscaling corrected discharge
    corrweights : None or DownScaleArray
        Initialized with None. Holds spatial distribution of correction weights (on 15 arcsec)
    correctiongrid : None or DownScaleArray
        Initialized with None. Holds correction values with are redistributed (dataset on 30 arcmin)

    """
    @Timer(name='decorator', text='Setting up the environment and reading necessary data takes {seconds:.0f} s')
    def __init__(self,
                 config,
                 **kwargs):

        with open(os.path.join(config.temp_dir, 'run_information.txt'), 'w') as f:
            temp = vars(config)
            for item in temp:
                f.write('{} : {}\n'.format(item, temp[item]))

        self.mode = config.mode
        self.dconfig = config
        kwargs.update(config.kwargs)
        self.kwargs = kwargs
        self.wg = WGData(self.dconfig, **kwargs)
        self.hydrosheds = HydroSHEDSData(self.dconfig, **kwargs)
        self.wg.calc_continentalarea_to_landarea_conversion_factor()
        self.wg.calc_surface_runoff_land_mm()
        if self.dconfig.mode == 'longterm_avg':
            self.wg.get_longterm_avg_version()
            self.wg.longterm_avg_converted = True
        self.daysinmonthdict = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

        self.tempdsarray = DownScaleArray(config, self.dconfig.aoi, **kwargs)
        self.surfacerunoffbaseddis = None
        self.correcteddis = None
        self.corrweights = None
        self.correctiongrid = None

    def prepare(self, staticdata=True, data=True, config=True):
        if self.dconfig.mode == 'ts':
            if staticdata:
                staticdata = {
                    'meanlfr': self.wg.land_fractions.data.reset_index().groupby('arcid')['landareafr'].mean(),
                    'wginput': self.wg.wginput.data,
                    'coords': self.wg.coords,
                    'flowacc': self.hydrosheds.flowacc,
                    'landratio_corr': self.hydrosheds.get_wg_corresponding_grid(self.wg.landratio_corr_path),
                    'largerivermask': self.hydrosheds.largerivermask(),
                    'cellpourpixel': self.hydrosheds.get_cellpourpixel(),
                    '30mingap_flowacc': self.hydrosheds.get_wg_corresponding_grid(self.wg.gap_flowacc_path),
                    'keepgrid': self.hydrosheds.keepGrid.copy(),
                    'shiftgrid': self.hydrosheds.shiftGrid.copy(),
                    'upstreampixelarea': self.hydrosheds.uparea,
                    'hydrosheds_geotrans': self.hydrosheds.hydrosheds_geotrans,
                    'pixelarea': self.hydrosheds.pixarea,
                    'globallakes_fraction': self.hydrosheds.globallakes_fraction_ar
                }
                if self.dconfig.l12harm:
                    staticdata['l12harmdata'] = self.hydrosheds.l12harmdata

                with open(os.path.join(self.dconfig.temp_dir, 'staticdata.pickle'), 'wb') as f:
                    pickle.dump(staticdata, f)

            if data:
                tasklist = []
                for yr in range(self.dconfig.startyear, self.dconfig.endyear+1):
                    for mon in range(1, 13):
                        data = {
                            'sr': self.wg.surface_runoff_land_mm.data.set_index(['arcid',
                                                                                 'month',
                                                                                 'year'])['variable'].loc[
                                slice(None), mon, yr],
                            'cellrunoffseries': self.wg.cell_runoff.set_index(['arcid',
                                                                               'month',
                                                                               'year'])['net_cell_runoff'].loc[
                                slice(None),
                                mon, yr],

                            'dis': self.wg.dis.data.set_index(['arcid', 'month', 'year'])['dis'].loc[
                                slice(None), mon, yr],
                            'gwrunoff': self.wg.gw_runoff.data.set_index(['arcid', 'month', 'year'])['variable'].loc[
                                slice(None), mon, yr],
                            'month': mon,
                            'year': yr,
                            'totalrunoff': self.wg.total_runoff.data.set_index(['arcid',
                                                                                'month',
                                                                                'year'])['variable'].loc[
                                slice(None), mon, yr],
                        }
                        if self.dconfig.correct_global_lakes:

                            data['gloaddition'] = self.wg.gloaddition.loc[slice(None), yr, mon]

                        tasklist.append(data)
                for i, task in enumerate(tasklist, 1):
                    with open('{}data_task{:03d}.pickle'.format(self.dconfig.temp_dir, i), 'wb') as f:
                        pickle.dump(task, f)

            if config:
                self.dconfig.pickle()

        del self.hydrosheds
        del self.wg


def run_prepared_downscaling(path, number_of_worker=2):

    with open(os.path.join(path, 'config.pickle'), 'rb') as f:
        config = pickle.load(f)
    pool = Pool(processes=number_of_worker)
    run_tasks = partial(run_task, path=path)
    poi_list = pool.map(run_tasks, [task for task in glob.glob('{}*task*.pickle'.format(path))])

    if isinstance(config.pois, pd.DataFrame):
        poidf = pd.DataFrame([x[1] for x in poi_list], index=[x[0] for x in poi_list]).sort_index()
        poidf.columns = config.pois['stationid'].to_list()
        poidf.to_csv(os.path.join(config.temp_dir ,
                                  '/selected_timeseries_data_{}_{}.csv'.format(config.startyear,
                                                                               config.endyear)
                                  )
                     )


def gather_finished_downscaling(path):
    with open(os.path.join(path , 'config.pickle'), 'rb') as f:
        config = pickle.load(f)
    poi_list = [pickle.load(open(result, 'rb')) for result in glob.glob('{}*result*.pickle'.format(path))]
    poi_list = [y for x in poi_list for y in x]
    poidf = pd.DataFrame([x[1] for x in poi_list], index=[x[0] for x in poi_list]).sort_index()
    poidf.columns = config.pois['stationid'].to_list()
    poidf.to_csv(os.path.join(path ,
                 'selected_timeseries_data_{}_{}.csv'.format(config.startyear,
                                                             config.endyear)
                              )
                 )
