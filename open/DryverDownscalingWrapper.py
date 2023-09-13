import pickle
import glob
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
from osgeo import gdal
from codetiming import Timer

from WGData import WGData
from HSData import HSData
from DownScaleArray import DownScaleArray
from DryverDownscaling import run_task


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
    mode : {'lta', 'ts'}
        mode of downscaling either lta = long term average or ts = timeseries
    dconfig : open.DryverDownscalingConfig.DownscalingConfig
        Configuratio object to run downscaling
    kwargs : dict
        kwargs updated with kwargs from DownscalingConfig
    wg : WGData.WGData
        data from WaterGAP
    hs : HSData.HSData
        data from HydroSHEDS
    daysinmonth : dict
        Dict with key = month as numeric and value = number of days
    tempdsarray : DownScaleArray
        Container for temporary results is able to write raster and pickle the data
    surfacerunoffbaseddis : None or DownScaleArray
        Initialized with None holds in downscaling the surface runoff based discharge
    correcteddis : None or DownScaleArray
        Initialized with None holds in downscaling corrected discharge
    corrweights : None or DownScaleArray
        Initialized with None holds spatial distribution of correction weights (on 15 arcsec)
    correctiongrid : None or DownScaleArray
        Initialized with None holds correction values with are redistributed (dataset on 30 arcmin)

    """
    @Timer(name='decorator', text='Setting up the environment and reading necessary data takes {seconds:.0f} s')
    def __init__(self,
                 config,
                 **kwargs):

        with open(config.temp_dir + 'run_information.txt', 'w') as f:
            temp = vars(config)
            for item in temp:
                f.write('{} : {}\n'.format(item, temp[item]))

        self.mode = config.mode
        self.dconfig = config
        kwargs.update(config.kwargs)
        self.kwargs = kwargs
        self.wg = WGData(self.dconfig, **kwargs)
        self.hs = HSData(self.dconfig, **kwargs)
        self.wg.calc_landmmconversion()
        self.wg.calc_surface_runoff_landmm()
        if self.dconfig.mode == 'lta':
            self.wg.get_lta_version()
            self.wg.lta_converted = True
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
                    'meanlfr': self.wg.landfractions.data.reset_index().groupby('arcid')['landareafr'].mean(),
                    'wginput': self.wg.wginput.data,
                    'coords': self.wg.coords,
                    'flowacc': self.hs.flowacc,
                    'landratio_corr': self.hs.get_wg_corespondend_grid(self.wg.landcorrpath),
                    'largerivermask': self.hs.largerivermask(),
                    'cellpourpixel': self.hs.get_cellpourpixel(),
                    '30mingapfa': self.hs.get_wg_corespondend_grid(self.wg.gapfapath),
                    'keepgrid': self.hs.keepGrid.copy(),
                    'shiftgrid': self.hs.shiftGrid.copy(),
                    'upstreampixelarea': self.hs.uparea,
                    'hsgt': self.hs.hsgt,
                    'pixelarea': self.hs.paar,
                    'glolakresfr': self.hs.glolakresfrar
                }
                if self.dconfig.l12harm:
                    staticdata['l12harmdata'] = self.hs.l12harmdata

                with open(self.dconfig.temp_dir + 'staticdata.pickle', 'wb') as f:
                    pickle.dump(staticdata, f)

            if data:
                tasklist = []
                for yr in range(self.dconfig.startyear, self.dconfig.endyear+1):
                    for mon in range(1, 13):
                        data = {
                            'sr': self.wg.surface_runoff_landmm.data.set_index(['arcid',
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
                        if self.dconfig.glolakredist:

                            data['gloaddition'] = self.wg.gloaddition.loc[slice(None), yr, mon]

                        tasklist.append(data)
                for i, task in enumerate(tasklist, 1):
                    with open('{}data_task{:03d}.pickle'.format(self.dconfig.temp_dir, i), 'wb') as f:
                        pickle.dump(task, f)

            if config:
                self.dconfig.pickle()

        del self.hs
        del self.wg


def run_prepared_downscaling(path, number_of_worker=2):

    with open(path + 'config.pickle', 'rb') as f:
        config = pickle.load(f)
    pool = Pool(processes=number_of_worker)
    run_tasks = partial(run_task, path=path)
    poi_list = pool.map(run_tasks, [task for task in glob.glob('{}*task*.pickle'.format(path))])

    if isinstance(config.pois, pd.DataFrame):
        poidf = pd.DataFrame([x[1] for x in poi_list], index=[x[0] for x in poi_list]).sort_index()
        poidf.columns = config.pois['stationid'].to_list()
        poidf.to_csv(config.temp_dir +
                     '/selected_timeseries_data_{}_{}.csv'.format(config.startyear,
                                                                  config.endyear))


def gather_finished_downscaling(path):
    with open(path + 'config.pickle', 'rb') as f:
        config = pickle.load(f)
    poi_list = [pickle.load(open(result, 'rb')) for result in glob.glob('{}*result*.pickle'.format(path))]
    poi_list = [y for x in poi_list for y in x]
    poidf = pd.DataFrame([x[1] for x in poi_list], index=[x[0] for x in poi_list]).sort_index()
    poidf.columns = config.pois['stationid'].to_list()
    poidf.to_csv(path +
                 'selected_timeseries_data_{}_{}.csv'.format(config.startyear,
                                                             config.endyear))
