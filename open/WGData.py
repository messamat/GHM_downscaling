import pickle
from os import path

import pandas as pd
import numpy as np
import osgeo.gdal
import osgeo.ogr
from osgeo.osr import SpatialReference
from codetiming import Timer

from open.WGReadingFunctions import read_variable, InputDir
"""
Needed general input data
../constants/xyarcid.csv


WGspecific data 
G_LAND_AREA_FRACTIONS_
G_CELL_RUNOFF_
G_SURFACE_RUNOFF_
G_GW_RUNOFF_
G_RIVER_AVAIL_
WaterGAPinputdir

"""

class WGData:
    """
    WaterGAP data which are read and processed during downscaling.

    Parameters
    ----------
    config : DownscalingConfig
        Downscaling configuration object
    **kwargs :
        keyword arguments

    Attributes
    ----------
    coords : pd.DataFrame
        pandas dataframe with x and y coordinate alongside with arcid
    read_variable_kwargs : dict
        dict to specify reading of WaterGAP files based on config
    aoi : ((float,float), (float,float))
        specified aoi if in kwargs else global aoi
    config : DownscalingConfig
        Downscaling configuration object
    """
    def __init__(self, config,
                 **kwargs):
        # CSV of longitude, latitude and corresponding ID for each cell in WaterGAP grids
        self.coords = pd.read_csv(path.join(path.dirname(__file__), '../constants/xyarcid.csv'))
        # Get analysis mode - long-term average or time.series mode
        self.mode = config.mode

        read_variable_kwargs = {
            "simoutput_path": config.wg_out_path,
            "timestep": 'month',
            "startyear": config.startyear,
            "endyear": config.endyear
        }

        #Subset analysis extent to "area_of_interest"
        if "area_of_interest" in kwargs:
            self.aoi = kwargs['area_of_interest']
            arcids = self.coords.set_index(["X", "Y"])
            relative_arcids_df = arcids.sort_index().loc[(slice(self.aoi[0][0], self.aoi[0][1]),
                                                     slice(self.aoi[1][0], self.aoi[1][1])), 'arcid']
            relative_arcids = relative_arcids_df.values.tolist()
            read_variable_kwargs['arcid_list'] = relative_arcids
            self.coords = relative_arcids_df.reset_index().set_index('arcid')
        else:
            self.aoi = ((-180., 180.), (-90., 90.))
            self.coords = self.coords.set_index('arcid')

        #Read data
        self.read_variable_kwargs = read_variable_kwargs
        self.wginput = InputDir().read(config.wg_in_path, explain=False, files=['G_FLOWDIR.UNF2', 'GR.UNF2',
                                                                                'GC.UNF2', 'GCONTFREQ.UNF0'],
                                       additional_files=False, cellarea=True)
        self.continentalarea_to_landarea_conversion_factor = None
        self.surface_runoff_land_mm = None
        self.land_fractions = read_variable(var='G_LAND_AREA_FRACTIONS_', **read_variable_kwargs)
        self.surface_runoff = read_variable(var='G_SURFACE_RUNOFF_', **read_variable_kwargs)
        self.total_runoff = read_variable(var='G_RUNOFF_', **read_variable_kwargs)
        self.gw_runoff = read_variable(var='G_GW_RUNOFF_mm_', **read_variable_kwargs)
        self.dis = read_variable(var='G_RIVER_AVAIL_', **read_variable_kwargs)  # km3 / month

        self.config = config
        self.longterm_avg_converted = False
        self.gap_flowacc_path = '{}{}_gap_flowacc.tif'.format(self.config.hydrosheds_path, config.continent)
        self.landratio_corr_path = self.config.hydrosheds_path + 'landratio_correction.tif'

        #Prepare data for redistribution of volume changes of global lake and reservoir
        if self.config.correct_global_lakes:
            cell_runoff = read_variable(var='G_CELL_RUNOFF_',
                                               **read_variable_kwargs).data.set_index(['arcid', 'year', 'month'])
            # cell runoff with extra treatment for global lakes and reservoirs
            modrkwargs = read_variable_kwargs.copy()
            modrkwargs['startyear'] = config.startyear - 1
            #Compute sum of lake and reservoir storage
            glolak = read_variable(var='G_GLO_LAKE_STORAGE_km3_', **modrkwargs).data
            dglolak = (glolak.set_index(['year', 'month', 'arcid']).unstack() -
                       glolak.set_index(['year', 'month', 'arcid']).unstack().shift(1)).stack()
            glores = read_variable(var='G_RES_STORAGE_km3_', **modrkwargs).data
            dglores = (glores.set_index(['year', 'month', 'arcid']).unstack() -
                       glores.set_index(['year', 'month', 'arcid']).unstack().shift(1)).stack()
            dgloresglolak = (dglores + dglolak).reset_index().set_index(['arcid', 'year', 'month']).sort_index()
            if 'arcid_list' in read_variable_kwargs:
                redistglwd = pd.read_csv(path.join(path.dirname(__file__),
                                                   '../constants/glolakandresredistribution.csv'))
                redistglwd = redistglwd.loc[[x in read_variable_kwargs['arcid_list'] for x in redistglwd.arcid], :]
            else:
                redistglwd = pd.read_csv(path.join(path.dirname(__file__),
                                                   '../constants/glolakandresredistribution.csv'))
            gloadditionlist = []
            for glwdunit in redistglwd.glwdunit.unique():
                redistmp = redistglwd[redistglwd.glwdunit == glwdunit].set_index('arcid')
                outflowcell = redistmp.outflowcell.iloc[0]
                if outflowcell not in read_variable_kwargs['arcid_list']:
                    continue
                gloaddition = dgloresglolak.loc[(outflowcell, slice(None), slice(None)), :].reset_index()
                gloaddition = gloaddition.loc[gloaddition.year >= config.startyear, 'variable'].values
                for aid in redistmp.index.get_level_values('arcid'):
                    frac = redistmp.loc[aid, 'fracarea']
                    if aid == outflowcell:
                        cell_runoff.loc[(aid, slice(None), slice(None)), 'net_cell_runoff'] = cell_runoff.loc[(aid, slice(None), slice(None)), 'net_cell_runoff'] + gloaddition

                    cell_runoff.loc[(aid, slice(None), slice(None)), 'net_cell_runoff'] = cell_runoff.loc[(aid, slice(None), slice(None)), 'net_cell_runoff'] - (gloaddition * frac)
                    years = [x for x in range(config.startyear, config.endyear+1)]
                    months = [x for x in range(1, 13)]
                    mix = pd.MultiIndex.from_product([[int(aid)], years, months], names=['arcid', 'years', 'month'])
                    gloadditionlist.append(pd.Series(gloaddition*frac, mix))

            self.cell_runoff = cell_runoff.reset_index()
            self.gloaddition = pd.concat(gloadditionlist)
        else:
            self.cell_runoff = read_variable(var='G_CELL_RUNOFF_', **read_variable_kwargs).data

    def get_longterm_avg_version(self):
        """
        Converts cell_runoff, surface_runoff, surface_runoff_land_mm, dis into long term average

        Returns
        -------
        None
        """
        self.cell_runoff.data = (self.cell_runoff.data.groupby(['arcid', 'year'])['net_cell_runoff'].sum().reset_index()
                                                      .groupby('arcid').mean().reset_index())
        self.surface_runoff.data = (self.surface_runoff.data.groupby(['arcid', 'year'])['variable'].sum().reset_index()
                                                            .groupby('arcid').mean().reset_index())
        self.surface_runoff_land_mm.data = (self.surface_runoff_land_mm.data.groupby(['arcid', 'year'])['variable']
                                                                          .sum().reset_index()
                                                                          .groupby('arcid').mean().reset_index())
        self.dis.data = (self.dis.data.groupby(['arcid', 'year'])['dis']
                                      .sum().reset_index()
                                      .groupby('arcid').mean().reset_index())
        self.longterm_avg_converted = True

    def calc_continentalarea_to_landarea_conversion_factor(self):
        """
        Calculates a continentalarea_to_landarea_conversion_factor.

        In cells where we have big swb surface runoff has to be converted to the landarea,
        thus we need a continentalarea_to_landarea_conversion_factor via landarea fraction / conintental area fraction

        Returns
        -------
        None
        """
        if self.continentalarea_to_landarea_conversion_factor is None:
            land_fractionswithcont = (self.land_fractions.data.set_index('arcid')
                                         .merge(self.wginput.data.loc[:, 'GCONTFREQ.UNF0'],
                                                left_index=True, right_index=True)
                                         .set_index(['year', 'month'], append=True))
            self.continentalarea_to_landarea_conversion_factor = (
                    land_fractionswithcont['landareafr'] / land_fractionswithcont['GCONTFREQ.UNF0'])

    def calc_surface_runoff_land_mm(self):
        """
        Converts surface runoff to surface runoff over land see calc_continentalarea_to_landarea_conversion_factor

        Returns
        -------
        None
        """
        if self.surface_runoff_land_mm is None:
            self.surface_runoff_land_mm = self.apply_continentalarea_to_landarea_conversion_factor(
                self.surface_runoff, self.continentalarea_to_landarea_conversion_factor)
            self.continentalarea_to_landarea_conversion_factor = None

    def convert_mm_to_km3(self, df):
        """
        Converts pd.DataFrame with WaterGAP data from mm to km3.

        Parameters
        ----------
        df : pd.DataFrame
            pd.DataFrame with WaterGAP data (mm to km3)

        Returns
        -------
        pd.DataFrame
        """
        a = df.merge(self.wginput.data.reset_index().loc[:, ['arcid', 'cellarea', 'GCONTFREQ.UNF0']],
                     left_on='arcid',
                     right_on='arcid')
        a['variable'] = a['variable'] * a['cellarea'] * a['GCONTFREQ.UNF0'] / 100. / 1000000
        return a.loc[:, df.columns]

    @staticmethod
    def apply_continentalarea_to_landarea_conversion_factor(wgdata, conversion):
        """
        Applies the conversion factor from mm in relation to continental area to mm in relation to land area.

        Parameters
        ----------
        wgdata : WaterGAPData
            data on which the conversion should be applied
        conversion :
            conversion factors
        Returns
        -------
        WaterGAPData
        """
        new = wgdata
        conversion.name = 'landm'
        tmp = wgdata.data
        if 'arcid' not in tmp.index.names or 'year' not in tmp.index.names or 'month' not in tmp.index.names:
            tmp = tmp.reset_index().set_index(['arcid', 'year', 'month'])
        tmp = tmp.merge(conversion, left_index=True, right_index=True)
        if wgdata.vars is None:
            vars_toconvert = ['variable']
        else:
            vars_toconvert = wgdata.vars
        for x in vars_toconvert:
            tmp[x] = tmp[x] / tmp['landm']
        new_cols = ['arcid', 'year', 'month']
        new_cols.extend(vars_toconvert)
        new.data = tmp.reset_index().loc[:, new_cols]
        return new

    @Timer(name='decorator')
    def arcid_six(self):
        """
        Creates a list with a spatial index for each arcid which can be calculated
        as row * 720 + col

        :return: list with spatial index
        """
        six = []
        for x in range(1, self.cell_runoff.nrows+1):
            arcidinput = self.wginput.data.loc[x]
            six.append(720 * arcidinput.loc['GR.UNF2'] + arcidinput.loc['GC.UNF2'])
        return six

    def get_flowdir(self, nan=-99):
        """
        receive a flow dir array

        Parameters
        ----------
        nan : int
            nan values

        Returns
        -------
        np.Array
        """
        return self.get_30min_array('flowdir', nan)

    def get_neighbours(self, **kwargs):
        """
        Calculates for each arcid a list with its neigbours can be used precalculated via kwargs

        Parameters
        ----------
        **kwargs : dict, optional
            keyword arguments

        Returns
        -------
        list of lists of int
            list per arcid with their neighbours
        """
        if 'neighb_cache' in kwargs:
            with open(kwargs['neighb_cache'], 'rb') as f:
                neighbourlist = pickle.load(f)
            return neighbourlist
        neighbourlist = []
        six = self.arcid_six()
        for x in range(self.cell_runoff.nrows):
            tmp_list = []
            if six[x]-1 in six:
                tmp_list.append(six.index(six[x]-1) + 1)
            if six[x]+1 in six:
                tmp_list.append(six.index(six[x]+1) + 1)
            if six[x]-720 in six:
                tmp_list.append(six.index(six[x]-720) + 1)
            if six[x]+720 in six:
                tmp_list.append(six.index(six[x]+720) + 1)
            if six[x]+719 in six:
                tmp_list.append(six.index(six[x]+719) + 1)
            if six[x]+721 in six:
                tmp_list.append(six.index(six[x]+721) + 1)
            if six[x]-719 in six:
                tmp_list.append(six.index(six[x]-719) + 1)
            if six[x]-721 in six:
                tmp_list.append(six.index(six[x]-721) + 1)
            neighbourlist.append(tuple(tmp_list))
        return neighbourlist

    def get_30min_array(self, s, nan=-99.):
        """
        Receive a numpy array in resolution of 30min(720 x 360 of WaterGAP data)

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
        if isinstance(s, pd.Series):
            df = self.wginput.data.merge(s, left_index=True, right_index=True)
            flowdir = False
        elif s == 'flowdir':
            df = self.wginput.data.rename(columns={"G_FLOWDIR.UNF2": "variable"})
            flowdir = True
        else:
            raise Exception('not implemented')
        for x in df.itertuples():
            array[x._2 - 1, x._3 - 1] = x.variable
        ar = array[360 - (self.aoi[1][1] + 90) * 2: 360 - (self.aoi[1][0] + 90) * 2,
                   (self.aoi[0][0] + 180) * 2: (self.aoi[0][1] + 180) * 2]
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
            # bottom botrder
            ar[-1, :][ar[-1, :] == 8] = 0
            ar[-1, :][ar[-1, :] == 4] = 0
            ar[-1, :][ar[-1, :] == 2] = 0
        return ar

    def create_inmemory_30min_pointds(self, inp, **kwargs):
        """
        Method which creates based on an WaterGAP resolution array (720*360) or a pandas Dataframe with arcid
         an inmemory point layer.

        Parameters
        ----------
        inp: {pd.Series, pd.DataFrame, np.array}
            pd.Dataframe or series with arcid as index or WaterGAP array (720*360)
        **kwargs : dict, optional
            keyword arguments

        Returns
        -------
        ogr point inmemory dataset

        """
        if isinstance(inp, (pd.Series, pd.DataFrame)):
            df = self.coords.merge(inp, left_index=True, right_index=True)
            inptype = 'pd'
        else:
            df = None
            inptype = 'other'
        drv = gdal.GetDriverByName('Memory')
        ds = drv.Create('runofftemp', 0, 0, 0, gdal.GDT_Unknown)
        lyr = ds.CreateLayer('runofftemp', None, ogr.wkbPoint)
        field_defn = ogr.FieldDefn('variable', ogr.OFTReal)
        lyr.CreateField(field_defn)
        if inptype == 'pd':
            for x in df.itertuples():
                feat = ogr.Feature(lyr.GetLayerDefn())
                feat.SetField("variable", x.variable)
                pt = ogr.Geometry(ogr.wkbPoint)
                pt.SetPoint(0, x.X, x.Y)
                feat.SetGeometry(pt)
                lyr.CreateFeature(feat)
                feat.Destroy()
        else:
            if 'all' in kwargs:
                for idx, value in np.ndenumerate(inp):
                    x = self.aoi[0][0] + (idx[1]/2) + 0.25
                    y = self.aoi[1][1] - ((idx[0]/2) + 0.25)
                    if not np.isnan(value):
                        feat = ogr.Feature(lyr.GetLayerDefn())
                        # irow, icol = self.wginput.data.loc[x.Index, ['GR.UNF2', 'GC.UNF2']]
                        feat.SetField("variable", value)
                        pt = ogr.Geometry(ogr.wkbPoint)
                        pt.SetPoint(0, x, y)
                        feat.SetGeometry(pt)
                        lyr.CreateFeature(feat)
                        feat.Destroy()
            else:
                for x in self.coords.itertuples():
                    col = int((x.X - self.aoi[0][0]) // 0.5)
                    row = int((self.aoi[1][1] - x.Y) // 0.5)
                    if not np.isnan(inp[row, col]):
                        feat = ogr.Feature(lyr.GetLayerDefn())
                        # irow, icol = self.wginput.data.loc[x.Index, ['GR.UNF2', 'GC.UNF2']]
                        feat.SetField("variable", inp[row, col])
                        pt = ogr.Geometry(ogr.wkbPoint)
                        pt.SetPoint(0, x.X, x.Y)
                        feat.SetGeometry(pt)
                        lyr.CreateFeature(feat)
                        feat.Destroy()
        return ds

    def interpol_to_30min(self, ds, **kwargs):
        return self.interpolation_to_grid(ds=ds,
                                          width=(self.aoi[0][1] - self.aoi[0][0]) * 2,
                                          height=(self.aoi[1][1] - self.aoi[1][0]) * 2,
                                          outputbounds=[self.aoi[0][0], self.aoi[1][1], self.aoi[0][1], self.aoi[1][0]],
                                          **kwargs)

    def interpol_to_6min(self, ds):
        return self.interpolation_to_grid(ds=ds,
                                          width=(self.aoi[0][1] - self.aoi[0][0]) * 10,
                                          height=(self.aoi[1][1] - self.aoi[1][0]) * 10,
                                          outputbounds=[self.aoi[0][0], self.aoi[1][1], self.aoi[0][1], self.aoi[1][0]])

    def interpol_to_15arcsec(self, ds):
        return self.interpolation_to_grid(ds=ds,
                                          width=(self.aoi[0][1] - self.aoi[0][0]) * 60 * 4,
                                          height=(self.aoi[1][1] - self.aoi[1][0]) * 60 * 4,
                                          outputbounds=[self.aoi[0][0], self.aoi[1][1], self.aoi[0][1], self.aoi[1][0]])

    @staticmethod
    def interpolation_to_grid(ds, width, height, outputbounds,
                              **kwargs):
        """ Takes inmemory point dataset and idw interpolates it to a grid

        if alg in kwargs this algorithm is used otherwise standard alg is used

        Parameters
        ----------
        ds : inmemory point dataset
            dataset which is interpolated
        width : float
            number of cells in x-ordinate
        height : float
            number cells in y-ordinate
        outputbounds : List[float]
            assigned output bounds: [ulx, uly, lrx, lry]
        **kwargs : dict, optional
            keyword arguments

        Returns
        -------
        GDAL raster
        """
        alg = "invdistnn:power=2.0:smoothing=0.0:radius=1.8:max_points=9:nodata=-99"
        if 'alg' in kwargs:
            alg = kwargs.pop('alg')
        out_raster_srs = SpatialReference()
        out_raster_srs.ImportFromEPSG(4326)
        go = gdal.GridOptions(format='MEM',
                              outputType=gdal.GDT_Float32,
                              layers='runofftemp',
                              zfield='variable',
                              outputSRS=out_raster_srs,
                              algorithm=alg,
                              width=width,
                              height=height,
                              outputBounds=outputbounds,
                              **kwargs)
        outr = gdal.Grid('outr', ds, options=go)
        return outr.ReadAsArray()

    @staticmethod
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
