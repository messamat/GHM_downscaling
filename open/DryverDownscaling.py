import warnings
import argparse
import glob
from functools import partial
import pickle

import numpy as np
import pandas as pd
import xarray as xr
from osgeo import gdal, ogr, osr
import numba
from numba import cfunc, carray
from numba.types import intc, CPointer, float64, intp, voidptr
from scipy.ndimage import generic_filter
from scipy import LowLevelCallable

from open.DownstreamGrid import get_inflow_sum, get_downstream_grid
from open.ModifiedFlowAcc import FlowAccTT
from open.DownScaleArray import DownScaleArray

def jit_filter_function(filter_function):
    """Decorator for use with scipy.ndimage.generic_filter."""
    jitted_function = numba.jit(filter_function, nopython=True)

    @cfunc(intc(CPointer(float64), intp, CPointer(float64), voidptr))
    def wrapped(values_ptr, len_values, result, data):
        values = carray(values_ptr, (len_values,), dtype=float64)
        result[0] = jitted_function(values)
        return 1
    return LowLevelCallable(wrapped.ctypes)


class DryverDownscaling:
    """
    Class that holds the algorithms and methods to conduct the downscaling with prepared data.

    """
    def __init__(self, datafn, staticdatafn, configfn):
        with open(datafn, 'rb') as f:
            self.data = pickle.load(f)
        with open(staticdatafn, 'rb') as f:
            self.staticdata = pickle.load(f)
        with open(configfn, 'rb') as f:
            self.dconfig = pickle.load(f)
        self.daysinmonthdict = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    def run_ts(self, month, year):
        """
        This method takes self (loaded and initalized DRYvER Downscaling object) to conduct the actual downscaling.


        **Principal approach of downscaling:**

        1. runoff based discharge

          a) 0.5 degree lr runoff (surface runoff or surface runoff + groundwater runoff) is disaggregated to 15 arc seconds HydroSHEDS river network

          b) this disaggregated runoff is routed through the HydroSHEDS river network via flow accumulation

        2. correction of runoffbased discharge

          a) the disaggregated runoff is compared as sum over one 0.5 grid cell (lr) to the discharge which is calculted for that cell (net cell runoff or discharge of cell - discharge of all upstream cells) to calculate a correction lr correction value

          b) the correction grid is used to correct the disaggregated runoff, the correction terms are calculated in a weighted way to apply the correction terms in streams with more runoff

        **1 a. The disaggregation from lr to hr:**

        - inverse distance interpolation from lr to intermediate resolution of 0.1 degree with following arguments power = 2, radius = 1.8 and max points = 9 , the intermediate resolution my be subject to smoothing methods (see later).
        - the intermediate resolution is dissaggregated to 15 arc seconds

        **1 b. flow accumulation:**

        The flow accumulation is done due to performance reasons with a own implementation of flow accumulation. See
        :class:`ModifiedFlowAcc.FlowAccTT`

        **Methods to improve and refine the disaggregation (1)**

        **Runoff smoothing**

        The (boolean) config parameter srsmoothing decides whether the input low resolution (:term:`lr`) runoff
        (:term:`sr` or :term:`sr` + :term:`gwr`) is smoothed. For that the methods
        :func:`~DryverDownscaling.DryverDownscaling.get_smoothed_runoff` ,
        :func:`~DryverDownscaling.DryverDownscaling.scipy_outl_removing` and
        :func:`~DryverDownscaling.DryverDownscaling.spatial_window_averaging`
        are used in specific parts of the disaggregation of lr runoff to hr runoff and discharge.
        First for all non-reliable (land area fractions below 0.5 %) and nan value grid cells are interpolated
        (get_smoothed_runoff). Then outliers are removed from the grid by limiting the value of a cell to two times the
        maximum or half of the minimum of its neighbours (scipy_outl_removing). After interpolating the 0.5 degree to
        0.1 degree the runoff smoothing is done with an averaging window of 5 (0.1 degree cells), which means that
        every cell is recalculated of itself and its surrounding 24 neighbours.

        **L12 harmonization**

        The dissaggregated runoff can be averaged over the small watersheds (Pfaffstetter level 12) using the
        HydroSHEDS boundaries based on the topological concept of the Pfaffstetter coding system. For this the boolean
        config parameter *l12harm* has to be true.

        **Precipitation correction**

        As precipitation is a dominant driving variable for the generation of runoff. It is possible to apply a
        precipitation correction which is calculated based on a monthly spatial variation taken from Worldclim V2
        data to include the subgrid heterogenity of precipitation patterns. Precipitation correction is activated, if
        the config parameter *preccor* is activated.

        **2 a. Calculation of correction :term:`lr` values**

        In order to calculate lr correction values the sum of hr dissaggregated runoff for each lr cell is compared to
        the discharge which is generated in this lr cell. (dis_in_cell_lr - sum_dissaggregated_runoff_hr)

        **2 b. The lr correction values are applied to the hr runoff**

        The lr correction values, which has been calculated in 2a are now applied in the cell but weighted depending
        on the streamflow which was calculated based on the dissaggregated runoff. In standard mode this is done by
        calculating the share of a hr cell on the total lr cell raw streamflow. This can weighting can be modified
        with a weighting factor which can be handed over in the configuration object with the keyword corrweightfactor
        (see also :func:`~DryverDownscaling.DryverDownscaling.get_corrweight`).

        Then the lr correction values are applied with the hr correction weights to the runoffbased discharge. In this
        process a limit for the correction values based on upstream area and removing artifacts mechanism can be
        activated (see also :func:`~DryverDownscaling.DryverDownscaling.calc_corrected_dis`)

        **Methods to improve the correction mechanisms (2)**

        **Large river correction**
        The hr river network includes more endorheic sinks then the lr river network due the resolution. Thus the
        correction values may not be sufficient. To account for this an additional adaption of the correction values
        are done. This is done if algorithms ref and ref_srplusgwr are chosen. For more information see
        :func:`~DryverDownscaling.DryverDownscaling.get_lrc_correctiongrid`

        **Partial shifting the correction grid**
        We don't know where in a lr grid cell the difference between runoff aggregated discharge and actual discharge
        origins from. Thus in order the lr correction values are partially shifted downstream in a weighted by upstream
        area mode. This leads to stronger corrections in larger streams.

        **Smoothing of the correction grid**
        In order to avoid positive-negative or vice versa sequence of lr correction values those sequence are removed
        in a max 10 times iterative process.


        :param month:
        :param year:
        :return:
        """
        #
        print('timestep {} {} started'.format(month, year))
        if self.dconfig.runoffsrc == 'cellrunoff':
            cellrunoff = self.data['cellrunoffseries'].dropna()

            cellrunoffm3s = (self.get_30min_array(cellrunoff, np.nan) / (self.daysinmonthdict[month]
                                                                         * 24 * 60 * 60) * 1000000000)
            cellrunoff15s = self.disaggregate(cellrunoffm3s, 120) / (120 * 120)
            cellrunoff15s[cellrunoff15s == -99] = np.nan
            cellrunoff15s = self.mask_wg_with_hydrosheds(cellrunoff15s)
            return self.flow_acc(cellrunoff15s)

        elif self.dconfig.runoffsrc == 'totalrunoff':
            sr = self.data['totalrunoff']
        elif self.dconfig.runoffsrc == 'srplusgwr':
            sr = self.data['sr'] + self.data['gwrunoff']
        elif self.dconfig.runoffsrc == 'sr':
            sr = self.data['sr']
        else:
            raise Exception('{} not implemented as runoffsrc'.format(self.dconfig.runoffsrc))

        if self.dconfig.srsmoothing:
            reliable_surfacerunoff = self.get_smoothed_runoff(sr)
        else:
            # no smoothing at all
            sr = sr.dropna()
            reliable_surfacerunoff = self.get_30min_array(
                sr, np.nan)
        del sr
        cellrunoffseries = self.data['cellrunoffseries']
        cellrunoffseries.name = 'variable'
        dis = self.data['dis']
        dis.name = 'variable'
        (dis_fg_conv) = self.get_runoff_based_dis(reliable_surfacerunoff, dis, cellrunoffseries,
                                                  month=month, yr=year)
        if not self.dconfig.discorr:
            return self.flow_acc(dis_fg_conv)
        del reliable_surfacerunoff
        # step8
        correctiongrid = self.calculate_lr_correctionvalues(dis_fg_conv, cellrunoffseries,
                                                            month=month, yr=year)
        del cellrunoffseries
        if 'corrweightfactor' in self.dconfig.kwargs:
            corrweights = self.get_corrweight(dis_fg_conv, self.dconfig.kwargs['corrweightfactor'])
        else:
            corrweights = self.get_corrweight(dis_fg_conv)

        if self.dconfig.largerivercorr:
            correcteddis = self.calc_corrected_dis(correctiongrid=correctiongrid,
                                                   corrweights=corrweights,
                                                   converted_runoff=dis_fg_conv,
                                                   month=month)
            dis = self.data['dis']
            dis.name = 'variable'
            correctiongrid = self.get_lrc_correctiongrid(dis,
                                                         correctiongrid=correctiongrid,
                                                         correcteddis=correcteddis,
                                                         month=month)
            del correcteddis

        if self.dconfig.corrgridshift:
            correctiongrid = self.shift_correctiongrid(correctiongrid)
        if self.dconfig.corrgridsmoothing:
            correctiongrid = self.smooth_correctiongrid(correctiongrid)
        correcteddis = self.calc_corrected_dis(correctiongrid=correctiongrid,
                                               corrweights=corrweights,
                                               converted_runoff=dis_fg_conv,
                                               threshold=self.dconfig.threshold,
                                               month=month)
        return correcteddis

    def save_and_run_ts(self):
        """
        Runs current timestep and saves results depending on chosen options.

        Returns
        -------
        timestep as month since 01-01-startyear, list with values of points of interest

        """
        month = self.data['month']
        year = self.data['year']
        result = self.run_ts(month, year)
        if isinstance(self.dconfig.pois, pd.DataFrame):
            ts_values = []
            for indexp, prow in self.dconfig.pois.iterrows():
                ts_values.append(result[prow['row'], prow['col']])
        else:
            ts_values = []

        if self.dconfig.write_result == 'raster':
            DownScaleArray(self.dconfig,
                           self.dconfig.aoi,
                           write_raster_trigger=True).load_data(result.astype(np.float32),
                                                                '15sec_dis_{}_{:02d}'.format(year, month))
        elif self.dconfig.write_result == 'nc':
            gt = self.staticdata['hydrosheds_geotrans']
            lon = [gt[0] + gt[1] * x for x in range(result.shape[1])]
            lat = [gt[3] - gt[1] * x for x in range(result.shape[0])]
            timestep = [month + (year - self.dconfig.startyear) * 12]
            ds = xr.DataArray(
                result.reshape((result.shape[0], result.shape[1], 1)),
                coords={
                    "lat": lat,
                    "lon": lon,
                    "time": timestep
                },
                dims=["lat", "lon", "time"]
            )
            ds.name = 'dis'

            ds['time'].attrs = {'standard_name': 'time',
                                'calendar': '360_day',
                                'units': 'months since {}-01-01'.format(self.dconfig.startyear)}
            ds.to_netcdf(self.dconfig.temp_dir +
                         '/15sec_dis_{}_{:02d}.nc4'.format(month, year),
                         encoding={'dis': {'zlib': True, 'complevel': 9,
                                           'dtype': 'float32'}},
                         unlimited_dims=['time'])
            del ds
            del result
        # self.wg.aoi = self.kwargs['area_of_interest']
        return month - 1 + (year - self.dconfig.startyear) * 12, ts_values

    def get_runoff_based_dis(self, reliablesurfacerunoff, lrdisseries, cellrunoffseries, **kwargs):
        """

        Parameters
        ----------
        reliablesurfacerunoff: np.array
            lr array with reliable runoff with which the discharge is calculated
        kwargs: dict
            keyword arguments which are handed over to convert runoff to dis

        Returns
        -------

        """
        if self.dconfig.srsmoothing:
            outlier_removed = self.scipy_outl_removing(reliablesurfacerunoff)
            tmp_ds = self.create_inmemory_30min_pointds(outlier_removed, all=True)
        else:
            tmp_ds = self.create_inmemory_30min_pointds(reliablesurfacerunoff, all=True)
        tmp_interp = self.interpolation_to_grid(tmp_ds, '6min')
        del tmp_ds
        tmp_interp[tmp_interp == -99] = np.nan
        if self.dconfig.srsmoothing:
            tmp_smooth = self.spatial_window_averaging(tmp_interp, 5)
        else:
            tmp_smooth = tmp_interp
        interpolated_smooth_15s = self.disaggregate(tmp_smooth, 24)
        del tmp_interp
        if self.dconfig.l12harm:
            masked_diss_tmp_smooth = self.harmonize_l12_hydrosheds(interpolated_smooth_15s)
        else:
            masked_diss_tmp_smooth = self.mask_wg_with_hydrosheds(interpolated_smooth_15s)
        del interpolated_smooth_15s
        conv = self.convert_runoff_to_dis(masked_diss_tmp_smooth, **kwargs)
        if self.dconfig.correct_global_lakes:
            # modification of reliable surfacerunoff with glolak and glores
            globallakes_fraction = self.staticdata['globallakes_fraction']
            gloaddition = self.disaggregate_smth(self.get_30min_array(self.data['gloaddition'] /
                                                                      (self.daysinmonthdict[kwargs['month']]
                                                                       * 24 * 60 * 60) * 1000000000, float(0)), 120)
            conv = conv - (globallakes_fraction * gloaddition)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                conv = np.where(conv < 0, 0, conv)
        return conv

    def get_smoothed_runoff(self, runoff):
        """

        This method takes in :term:`lr` runoff (may be :term:`sr` or :term:`sr` + :term:`gwr`) and first drops grid
        cells, where the land area fraction is below 0.5 %. Then those droped grid cells and also grid cells which had
        no values are filled with an interpolated values.

        Parameters
        ----------
        runoff : pd.Series
            runoff lr which has the information of the GHM

        Returns
        -------
        np.array
            an array holding the smoothed surface and interpolated missing values

        """


        # STEP4a
        # find out cells which need to be replaced
        # PRELIMINIARY
        meanlfr = self.staticdata['meanlfr']
        not_reliable_arcids = meanlfr[meanlfr <= 0.5].index
        reliable_surface_runoff = (runoff.drop(not_reliable_arcids))
        upper = self.get_30min_array(reliable_surface_runoff, np.nan)
        points = self.create_inmemory_30min_pointds(upper)
        lower = self.interpolation_to_grid(points, '30min',
                                           alg="invdistnn:power=1.0:smoothing=0.0:radius=20"
                                               ":min_points=1:max_points=9:nodata=-99")
        new_surface_runoff_land_mm = self.stack(upper, lower)
        return new_surface_runoff_land_mm

    def get_30min_array(self, s, nan=-99):
        array = np.full((360, 720), nan)
        wginput = self.staticdata['wginput']
        aoi = self.dconfig.aoi
        if isinstance(s, pd.Series):
            s.name = 'variable'
            df = wginput.merge(s, left_index=True, right_index=True)
            flowdir = False
        elif s == 'flowdir':
            df = wginput.rename(columns={"G_FLOWDIR.UNF2": "variable"})
            flowdir = True
        else:
            raise Exception('not implemented')
        for x in df.itertuples():
            array[x._2 - 1, x._3 - 1] = x.variable
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
        aoi = self.dconfig.aoi
        coords = self.staticdata['coords']
        if isinstance(inp, (pd.Series, pd.DataFrame)):
            df = coords.merge(inp, left_index=True, right_index=True)
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
                    x = aoi[0][0] + (idx[1] / 2) + 0.25
                    y = aoi[1][1] - ((idx[0] / 2) + 0.25)
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
                for x in coords.itertuples():
                    col = int((x.X - aoi[0][0]) // 0.5)
                    row = int((aoi[1][1] - x.Y) // 0.5)
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

    def interpolation_to_grid(self, ds, resolution, **kwargs):
        aoi = self.dconfig.aoi
        if resolution == '6min':
            width = (aoi[0][1] - aoi[0][0]) * 10
            height = (aoi[1][1] - aoi[1][0]) * 10
        elif resolution == '30min':
            width = (aoi[0][1] - aoi[0][0]) * 2
            height = (aoi[1][1] - aoi[1][0]) * 2
        elif resolution == '15sec':
            width = (aoi[0][1] - aoi[0][0]) * 60 * 4
            height = (aoi[1][1] - aoi[1][0]) * 60 * 4
        else:
            raise Exception('interpolation not implemented')
        outputbounds = [aoi[0][0], aoi[1][1], aoi[0][1], aoi[1][0]]
        alg = "invdistnn:power=2.0:smoothing=0.0:radius=1.8:max_points=9:nodata=-99"
        if 'alg' in kwargs:
            alg = kwargs.pop('alg')
        out_raster_srs = osr.SpatialReference()
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
        res = outr.ReadAsArray().copy()
        del outr
        return res

    def mask_wg_with_hydrosheds(self, wg):
        """
        Mask dissaggregated a DownScaleArray with a Hydrosheds rasterfile like flow directions.

        This process masks the 15sec from WaterGAP originating data (DownScaleArray) with Hydrosheds data. They must not
        be the same size but DownScaleArray must be at least the size of Hydrosheds array resp. raster. The
        DownScaleArray is then clipped to Hydrosheds array and grid cells which are nan in Hydrosheds array are set nan
        in DownScaleArray.

        :param wg: DownScale Array with 15sec data originating from WaterGAP data
        :type wg: DownScaleArray
        :return: masked and clipped DownScaleArray
        :rtype: np.array
        """
        hydrosheds_ar = self.staticdata['pixelarea']
        array = np.full(hydrosheds_ar.shape, np.nan)
        hydrosheds_geotrans = self.staticdata['hydrosheds_geotrans']
        coloffset = int(round(self.dconfig.aoi[0][0] - hydrosheds_geotrans[0]) // 0.5 * 120 * -1)
        rowoffset = int(round(self.dconfig.aoi[1][1] - hydrosheds_geotrans[3]) // 0.5 * 120)
        offset = wg[rowoffset:, coloffset:]
        rowix = array.shape[0] - offset.shape[0]
        colix = array.shape[1] - offset.shape[1]
        if rowix == 0:
            rowix = array.shape[0]
        if colix == 0:
            colix = array.shape[1]
        wgdata = offset[:rowix,
                        :colix]
        array[~np.isnan(hydrosheds_ar)] = wgdata[~np.isnan(hydrosheds_ar)]
        # Setting new area of interest as array is clipped
        # wg.aoi = ((round(hydrosheds_geotrans[0]), round(hydrosheds_geotrans[0]) + array.shape[1]//240),
                  # (round(hydrosheds_geotrans[3] - array.shape[0] // 240), round(hydrosheds_geotrans[3])))
        # self.wg.aoi = wg.aoi
        return array

    def harmonize_l12_hydrosheds(self, ar):
        oldflat = ar.flatten()
        newflat = oldflat.copy()
        newflat[:] = np.nan
        for basin in self.staticdata['l12harmdata']:
            newflat[basin] = np.mean(oldflat[basin])
        return newflat.reshape(ar.shape)

    def flow_acc(self, dis15s):
        newar = self.staticdata['flowacc'].get(dis15s)
        newar[np.isnan(self.staticdata['pixelarea'])] = np.nan
        return newar

    def calculate_lr_correctionvalues(self, dis_fg_conv, cellrunoffseries, **kwargs):
        # step 8a
        # reaggregation of step5disconv
        dis_pix_30min = self.aggsum(dis_fg_conv, 120, zeroremove=False)
        if self.dconfig.mode == 'ts':
            cellrunoffm3s = (self.get_30min_array(cellrunoffseries, np.nan) / (self.daysinmonthdict[kwargs['month']]
                             * 24 * 60 * 60) * 1000000000)
        else:
            cellrunoffm3s = self.get_30min_array(cellrunoffseries, np.nan) / (365 * 24 * 60 * 60) * 1000000000
        landratio_corr = self.staticdata['landratio_corr']
        cellrunoffm3s *= landratio_corr
        cellrunoffm3s = self.stack(upper=cellrunoffm3s, lower=dis_pix_30min)
        dif_dis_30min = cellrunoffm3s - dis_pix_30min
        return dif_dis_30min

    def get_corrweight(self, converted_runoff, weightingfactor=1):
        """
        The correction weights are calculated based on discharge calculated with raw runoff with a weighting factor.

        The weighting factor is a multiplicator for the weighting. This means with a weighting factor 2 a hr cell which
        holds 20% of the runoff based discharge of the lr cell gets a correction weight of 0.4 instead of 0.2 with
        weighting factor 1.

        Parameters
        ----------
        runoffbaseddis: np.array
            array with runoff based discharge in lr
        weightingfactor: number, default 1
            weightingfactor which determines with which factor the runoffbased discharge should be representative

        Returns
        -------
        np.array
            hr grid with correction weights, per lr grid cell sum up to one

        """
        # Step 8b

        runoffbaseddis = self.flow_acc(converted_runoff)
        dis_fg_sum_30min = self.aggsum(runoffbaseddis, 120, zeroremove=False)
        dis_fg_sum_30min[dis_fg_sum_30min == 0] = 1
        dis_corr_weight = runoffbaseddis / self.disaggregate_smth(dis_fg_sum_30min, 120)
        if weightingfactor != 1:
            dis_corr_weight = ((dis_corr_weight - (1 / (120 * 120))) * weightingfactor) + (1 / (120 * 120))
        return dis_corr_weight

    def calc_corrected_dis(self, correctiongrid, corrweights, converted_runoff,
                           threshold=0.001, **kwargs):
        """
        Based on the raw discharge calculated with dissaggregated runoff, a lr correction grid and correction weights
        the downscaled discharge is calculated.

        The correction grid is dissaggregated to hr and multiplied with correction weights. Then the threshold value is
        applied if parameter apply_thresh == True. This threshold is a multiplicator for upstreamarea, which is then
        used as maximum correction value in positive or negative direction. Then the weighted and optional limited
        correction values are then applied on the raw discharge calculated with dissaggregated runoff. To avoid
        implausible artifacts of the correction. Negative discharge values are set to zero.

        Parameters
        ----------
        correctiongrid: np.array
            lr grid with correction values
        corrweights: np.array
            hr grid with weights for the lr correction values
        converted_runoff: np.array
            grid with converted_runoff
        threshold: float or None
            multiplicator for upstream area to limit hr correction values, None if no threshold should be applied

        Returns
        -------
        np.array
            hr grid with corrected discharge

        """
        dis_pix_corr_15s = self.disaggregate_smth(correctiongrid.data, 120) * corrweights.data

        if threshold is not None:
            ctar = self.staticdata['upstreampixelarea'] * threshold
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                negative_corr = np.where(dis_pix_corr_15s < 0, -1, 1)
            dis_pix_corr_15s = np.minimum(np.abs(dis_pix_corr_15s), ctar)
            dis_pix_corr_15s = dis_pix_corr_15s * negative_corr
            del negative_corr

        cor_dis = converted_runoff + dis_pix_corr_15s

        corrected_dis = self.flow_acc(cor_dis)
        # manual correction of negative discharge
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            corrected_dis[corrected_dis < 0] = 0
        return corrected_dis

    def get_lrc_correctiongrid(self, dis, correcteddis, correctiongrid, **kwargs):
        """
        This method adapts the correction grid to account for differences in river networks (i.e. missing endorheic
        sinks). This adaption is only done in large rivers ( geq 50000 km2)

        Parameters
        ----------
        dis
        correcteddis
        correctiongrid
        kwargs

        Returns
        -------

        """

        fd = self.get_30min_array('flowdir')
        lrivermask = self.staticdata['largerivermask']
        if self.dconfig.mode == 'ts':
            dism3s = self.get_30min_array(dis, np.nan) / (self.daysinmonthdict[kwargs['month']] * 24 * 60
                                                          * 60) * 1000000000
        else:
            dism3s = self.get_30min_array(dis, np.nan) / (365 * 24 * 60 * 60) * 1000000000
        dis_largerivers_wg_30min = dism3s * lrivermask
        cell_dis_contribution_wg = dis_largerivers_wg_30min - get_inflow_sum(dis_largerivers_wg_30min, fd)
        cellpourpixel = self.staticdata['cellpourpixel']
        correcteddis = correcteddis * cellpourpixel
        tmp_maxaccudis30min = self.aggmax(correcteddis, 120)
        dis_largerivers_hydrosheds_30min = tmp_maxaccudis30min * lrivermask
        cell_dis_contribution_hydrosheds = dis_largerivers_hydrosheds_30min - get_inflow_sum(dis_largerivers_hydrosheds_30min,
                                                                             fd)
        cell_dis_contribution_dif = cell_dis_contribution_wg - cell_dis_contribution_hydrosheds
        gapmask = 1 - lrivermask
        transfer_value_grid = gapmask * cell_dis_contribution_dif
        transfer_ddm_grid = fd * (1 - lrivermask)
        transfer_ddm_grid[fd == -99] = -99
        fa = self.staticdata['30mingap_flowacc']
        transfer_accu_grid = FlowAccTT(transfer_ddm_grid, fa, True).get(transfer_value_grid,
                                                                        no_negative_accumulation=False)
        new_diff_dis_30min = correctiongrid + ((cell_dis_contribution_dif + transfer_accu_grid)
                                               * lrivermask)

        return new_diff_dis_30min

    def shift_correctiongrid(self, correctiongrid):
        fd30min = self.get_30min_array('flowdir')
        corr_grid = ((correctiongrid * self.staticdata['keepgrid']) +
                     get_inflow_sum((correctiongrid * self.staticdata['shiftgrid']), fd30min))
        return corr_grid

    def smooth_correctiongrid(self, correctiongrid):
        fd30min = self.get_30min_array('flowdir')
        corr_grid = correctiongrid
        for i in range(10):
            down_corr_grid = get_downstream_grid(corr_grid, fd30min, None)
            min_diff_grid = np.min([np.abs(corr_grid), np.abs(down_corr_grid)], axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                sign_grid = (((corr_grid < 0) * (down_corr_grid > 0)).astype(int) -
                             ((corr_grid > 0) * (down_corr_grid < 0)).astype(int))
            change_grid = sign_grid * min_diff_grid
            inflow_change_grid = get_inflow_sum(change_grid, fd30min)
            corr_grid = corr_grid + change_grid - inflow_change_grid

        return corr_grid

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

    def convert_runoff_to_dis(self, runoff, **kwargs):
        #TODO rename method
        if self.dconfig.mode == 'ts':
            divisionterm = (self.daysinmonthdict[kwargs['month']] * 24 * 60 * 60)
        elif self.dconfig.mode == 'longterm_avg':
            divisionterm = (365 * 24 * 60 * 60)
        else:
            raise Exception()
        area = self.staticdata['pixelarea']
        data = (runoff * area
                * 1000 / divisionterm)
        return data

    @staticmethod
    def scipy_outl_removing(ar):
        """
        Remove outliers (2 times max of neighbours or half of min of neighbours)

        Parameters
        ----------
        ar : np.array

        Returns
        -------
        np.array
            input array with removed outliers

        """
        def help_outl_remove(x):
            value = x[4]
            neighs = x[[0, 1, 2, 3, 5, 6, 7, 8]]
            if np.isnan(x[4]):
                return np.nan
            elif sum(np.isnan(neighs)) == 8:
                return value
            elif np.nanmin(neighs)/2 < value < np.nanmax(neighs)*2:
                return value
            elif np.nanmin(neighs)/2 > value:
                return np.nanmin(neighs)/2
            else:
                return np.nanmax(neighs)*2
        ar[ar == -99] = np.nan
        ar = generic_filter(ar, help_outl_remove, 3)
        return ar

    @staticmethod
    def spatial_window_averaging(ar, windowsize):
        """
        Applies spatial window mean on array

        Parameters
        ----------
        ar : np.array
            array which is smoothed
        windowsize : int
            size of the window on which basis the array is smoothed ( averaged)

        Returns
        -------
        np.array
            smoothed array
        """
        ar[np.isnan(ar)] = -9999

        @jit_filter_function
        def nanmean(values):
            result = 0.
            i = 0
            for v in values:
                if v >= 0:
                    result += v
                    i += 1
            if i > 0:
                return result/i
            else:
                return np.nan

        ar = generic_filter(ar, nanmean, windowsize)
        return ar

    @staticmethod
    def disaggregate(dsarray, factor):
        """
        Disaggregation of an array into higher resolution

        Parameters
        ----------
        dsarray : DownscalingArray
            array which should be dissaggregated into higher resolution
        factor : number
            target resolution as string
        Returns
        -------
        np.Array
        """
        a = np.repeat(dsarray.data, factor, axis=0)
        b = np.repeat(a, factor, axis=1)
        return b

    @staticmethod
    def aggsum(what, factor, zeroremove=True):
        what = what.copy()
        what[np.isnan(what)] = 0
        result = np.einsum('ijkl->ik',
                           what.reshape(what.shape[0] // factor, factor, -1, factor))
        if zeroremove:
            result[result == 0] = np.nan
        return result

    @staticmethod
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

    @staticmethod
    def disaggregate_smth(what, factor):
        """ Takes a 2D numpy array and repeat the value of one pixel by factor along both axis

        :param what: 2D Array which should be dissaggregated
        :type what: np.ndarray
        :param factor: how often the value is repeated
        :type factor: int
        :return: disaggregated 2D array which values are repeated by the factor
        :rtype: np.ndarray
        """
        a = np.repeat(what, factor, axis=0)
        b = np.repeat(a, factor, axis=1)
        return b


def run_task(task, path):
    dd = DryverDownscaling(datafn=task, staticdatafn=path + 'staticdata.pickle', configfn=path + 'config.pickle')
    return dd.save_and_run_ts()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run downscaling for one TS')
    parser.add_argument('Taskpart', metavar='T', type=int, help="which part should be worked on")
    parser.add_argument('path', metavar='path', type=str, help="the path where where the downscaling should happen")
    parser.add_argument('taskn', type=int, help="in how many groups the subtasks should be distributed")
    args = parser.parse_args()

    run_tasks = partial(run_task, path=args.path)
    tasklist = [task for task in glob.glob('{}*task*.pickle'.format(args.path))]
    splitlist = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
    splitfactor = (-(-len(tasklist) // args.taskn))
    tasklist = splitlist(tasklist, splitfactor)
    poi_list = [run_tasks(x) for x in tasklist[args.Taskpart-1]]

    with open('{}result_part{:02d}.pickle'.format(args.path, args.Taskpart), 'wb') as f:
        pickle.dump(poi_list, f)
