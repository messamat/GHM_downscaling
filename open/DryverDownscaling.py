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

def jit_filter_function(filter_function):
    """Decorator for use with scipy.ndimage.generic_filter.
    The point is to optimize the function with Numba, which generates optimized machine code from pure Python code.
    jit stands for 'just-in-time' compilation.
    Numba reads the Python bytecode for a decorated function and combines this with information about the types of the
    input arguments to the function. It analyzes and optimizes the code, and finally uses a compiler library
    to generate a machine code version of the function, tailored to the CPU capabilities. This compiled version is
    then used every time the function is called."""

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
    def __init__(self, in_taskdata_dict_picklepath, in_staticdata_dict_picklepath, in_config_dict_picklepath):
        """
        
        Parameters
        ----------
        in_taskdata_dict_picklepath: path to pickle containing input formatted WG data for that task (time step)
                                 -> Implemented/created with DryverDownscalingWrapper.prepare() in RunDownscaling.py
        in_staticdata_dict_picklepath: path to pickle containing input static data (e.g., HydroSHEDS flow acc,
                                        land areas, upstream pixel area)
                                 -> Implemented/created with DryverDownscalingWrapper.prepare() in RunDownscaling.py
        in_config_dict_picklepath: path to pickle containing configuration parameters to run downscaling
                                 -> Implemented/created with DownscalingConfig in RunDownscaling.py
        """

        with open(in_taskdata_dict_picklepath, 'rb') as f:
            self.taskdata = pickle.load(f)
        with open(in_staticdata_dict_picklepath, 'rb') as f:
            self.staticdata = pickle.load(f)
        with open(in_config_dict_picklepath, 'rb') as f:
            self.dconfig = pickle.load(f)
        self.daysinmonth_dict = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                                 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}

    def run_ts(self, month, year):
        """
        This method takes self (loaded and initialized DRYvER Downscaling object) to conduct the actual downscaling.


        **Principal approach of downscaling:**

        1. runoff based discharge

          a) 0.5 degree lr runoff (surface runoff or surface runoff + groundwater runoff) is disaggregated to 15 arc
          seconds HydroSHEDS river network

          b) this disaggregated runoff is routed through the HydroSHEDS river network via flow accumulation

        2. correction of runoff-based discharge

          a) the disaggregated runoff is compared as sum over one 0.5 grid cell (lr) to the discharge which is
          calculated for that cell (net cell runoff or discharge of cell - discharge of all upstream cells) to calculate
           a correction lr correction value

          b) the correction grid is used to correct the disaggregated runoff, the correction terms are calculated in a
          weighted way to apply the correction terms in streams with more runoff

        **1 a. The disaggregation from lr to hr:**

        - inverse distance interpolation from lr to intermediate resolution of 0.1 degree with the following arguments:
        power = 2, radius = 1.8 and max points = 9 , the intermediate resolution may be subject to smoothing methods
        (see later).

        - the intermediate resolution is disaggregated to 15 arc seconds

        **1 b. flow accumulation:**

        The flow accumulation is done due to performance reasons with a own implementation of flow accumulation. See
        :class:`ModifiedFlowAcc.FlowAccTT`

        **Methods to improve and refine the disaggregation (1)**

        **Runoff smoothing**

        The (boolean) config parameter sr_smoothing decides whether the input low resolution (:term:`lr`) runoff
        (:term:`sr` or :term:`sr` + :term:`gwr`) is smoothed. For that the methods
        :func:`~DryverDownscaling.DryverDownscaling.get_smoothed_runoff` ,
        :func:`~DryverDownscaling.DryverDownscaling.remove_outliers_scipy` and
        :func:`~DryverDownscaling.DryverDownscaling.spatial_window_averaging`
        are used in specific parts of the disaggregation of lr runoff to hr runoff and discharge.
        First, for all non-reliable (land area fractions below 0.5 %) and nan value grid cells are interpolated
        (get_smoothed_runoff). Then, outliers are removed from the grid by limiting the value of a cell to two times the
        maximum or half of the minimum of its neighbours (remove_outliers_scipy). After interpolating the 0.5 degree to
        0.1 degree the runoff smoothing is done with an averaging window of 5 (0.1 degree cells), which means that
        every cell is recalculated of itself and its surrounding 24 neighbours.

        **L12 harmonization**

        The disaggregated runoff can be averaged over the small watersheds (Pfaffstetter level 12) using the
        HydroSHEDS boundaries based on the topological concept of the Pfaffstetter coding system. For this the boolean
        config parameter *l12harm* has to be true.

        **Precipitation correction**

        As precipitation is a dominant driving variable for the generation of runoff. It is possible to apply a
        precipitation correction which is calculated based on a monthly spatial variation taken from Worldclim V2
        data to include the subgrid heterogeneity of precipitation patterns. Precipitation correction is activated, if
        the config parameter *preccor* is activated.

        **2 a. Calculation of correction :term:`lr` values**

        In order to calculate lr correction values the sum of hr disaggregated runoff for each lr cell is compared to
        the discharge which is generated in this lr cell. (dis_in_cell_lr - sum_disaggregated_runoff_hr)

        **2 b. The lr correction values are applied to the hr runoff**

        The lr correction values, which has been calculated in 2a are now applied in the cell but weighted depending
        on the streamflow which was calculated based on the disaggregated runoff. In standard mode this is done by
        calculating the share of a hr cell on the total lr cell raw streamflow. This can weighting can be modified
        with a weighting factor which can be handed over in the configuration object with the keyword corrweightfactor
        (see also :func:`~DryverDownscaling.DryverDownscaling.get_corrweight`).

        Then the lr correction values are applied with the hr correction weights to the runoffbased discharge. In this
        process a limit for the correction values based on upstream area and removing artifacts mechanism can be
        activated (see also :func:`~DryverDownscaling.DryverDownscaling.correct_dis`)

        **Methods to improve the correction mechanisms (2)**

        **Large river correction**
        The hr river network includes more endorheic sinks than the lr river network due the resolution. Thus the
        correction values may not be sufficient. To account for this an additional adaption of the correction values
        are done. This is done if algorithms ref and ref_srplusgwr are chosen. For more information see
        :func:`~DryverDownscaling.DryverDownscaling.get_largerivers_correction_grid`

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
        #Get and pre-format WG data to process -------------------------------------------------------------------------
        if self.dconfig.runoff_src == 'cellrunoff': #This is not currently applied
            cellrunoff = self.taskdata['netdis_30min_series'].dropna()

            #Convert cell runoff from km3/yr to m3/s
            cellrunoff_m3s = (self.get_30min_array(cellrunoff, np.nan) / (self.daysinmonth_dict[month]
                                                                          * 24 * 60 * 60) * 1000000000)

            #Disaggregate from 30 min (HR) to 15s (LR), assigning the same cell runoff to each HR cell
            # (a 14400th of that in the LR cell)
            cellrunoff_15s = self.disaggregate(cellrunoff_m3s, 120) / (120 * 120)
            cellrunoff_15s[cellrunoff_15s == -99] = np.nan

            #Remove HR cells where HydroSHEDS pixel area raster is NoData
            cellrunoff_15s = self.mask_wg_with_hydrosheds(cellrunoff_15s)

            #Compute flow accumulation on cell runoff (i.e., runoff-based discharge)
            return self.flow_acc(cellrunoff_15s)
            #->  end of first phase of principal downscaling

        elif self.dconfig.runoff_src == 'totalrunoff':
            sr = self.taskdata['totalrunoff']
        elif self.dconfig.runoff_src == 'srplusgwr':
            sr = self.taskdata['sr'] + self.taskdata['gwrunoff']
        elif self.dconfig.runoff_src == 'sr':
            sr = self.taskdata['sr']
        else:
            raise Exception('{} not implemented as runoff_src'.format(self.dconfig.runoff_src))

        #Interpolate or drop NA cells (and cells with < 0.5% land fraction)
        if self.dconfig.sr_interp_wg_nas: #Not turned on right now - need to check if "get_smoothed_runoff" function works
            #Interpolate all LR cells with land area fractions below 0.5 % and nan value grid cells
            reliable_surfacerunoff_ar = self.get_smoothed_runoff(sr)
        else:
            #Remove na value grid cells
            sr = sr.dropna()
            reliable_surfacerunoff_ar = self.get_30min_array(sr, np.nan)
        del sr

        #--------- Compute initial 15-sec runoff-based cell discharge ---------------------------------------------------
        # if sr.smoothing == True: remove outliers
        # inverse distance interpolation from lr to intermediate resolution of 0.1 degree
        # if sr.smoothing == True: perform 5x5 mean filtering on 6 min raster (excluding NAs)
        # disaggregate the 6 min raster to 15 arc seconds
        # mask with HydroSHEDS reference layer (remove 15-sec cells where original HydroSHEDS pixel area raster is NoData)
        # compute runoff-based discharge
        # correct runoff-based discharge for changes in storage of global lakes and reservoirs
        # -> yields a 15 sec array of runoff-based discharge (in m3/s) generated in each cell (i.e., non-accumulated)
        (runoffbased_celldis_15s_ar) = self.get_runoff_based_celldis(reliable_surfacerunoff_ar, month=month, yr=year)
        
        #If not implementing further discharge correction, return this flow-accumulated runoff-based cell discharge  ---
        if not self.dconfig.dis_corr:
            return self.flow_acc(runoffbased_celldis_15s_ar)
        del reliable_surfacerunoff_ar
        
        #Compute correction values -------------------------------------------------------------------------------------
        #Compute initial correction values in each 30-min cell
        #the sum of disaggregated runoff-based discharge from all HR cells within each LR cell is compared to the net
        # discharge which is generated in this LR cell (this serves to account for additional information from the
        # routing routine of the LR GHM, in particular about the impact of surface water bodies and human water use on
        # streamflow).
        netdis_30min_series = self.taskdata['netdis_30min_series']
        netdis_30min_series.name = 'variable'

        correction_grid_30min = self.calculate_lr_correctionvalues(runoffbased_celldis_15s_ar=runoffbased_celldis_15s_ar,
                                                                   netdis_30min_series=netdis_30min_series,
                                                                   month=month, yr=year)
        del netdis_30min_series

        #Compute what proportion of the 30-min correction value should apply to each 15-sec cell
        if 'corrweightfactor' in self.dconfig.kwargs:
            correction_weights_15s = self.get_corrweight(runoffbased_celldis_15s_ar, self.dconfig.kwargs['corrweightfactor'])
        else:
            correction_weights_15s = self.get_corrweight(runoffbased_celldis_15s_ar)

        #Change correction values in 30-min cells to apply an even greater proportion of the correction on large rivers
        # (if large_river_corr is True)
        if self.dconfig.large_river_corr:
            corrected_dis_15s = self.correct_dis(correction_grid_30min=correction_grid_30min,
                                                 correction_weights_15s=correction_weights_15s,
                                                 runoffbased_celldis_15s_ar=runoffbased_celldis_15s_ar,
                                                 month=month)
            wg_dis_30min = self.taskdata['dis']
            wg_dis_30min.name = 'variable'
            correction_grid_30min = self.get_largerivers_correction_grid(
                wgdis_series_30min=wg_dis_30min,
                correction_grid_30min=correction_grid_30min,
                precorrected_dis=corrected_dis_15s,
                month=month)
            del corrected_dis_15s

        #Shift and smooth correction values downstream at 30-min
        if self.dconfig.corr_grid_shift:
            correction_grid_30min = self.shift_correction_grid(correction_grid_30min)
        if self.dconfig.corr_grid_smoothing:
            correction_grid_30min = self.smooth_correction_grid(correction_grid_30min)

        #Apply correction at 15-sec and accumulate corrected net discharge downstream ----------------------------------
        corrected_dis_15s = self.correct_dis(correction_grid_30min=correction_grid_30min,
                                             correction_weights_15s=correction_weights_15s,
                                             runoffbased_celldis_15s_ar=runoffbased_celldis_15s_ar,
                                             correction_threshold=self.dconfig.correction_threshold,
                                             month=month)
        return corrected_dis_15s

    def save_and_run_ts(self):
        """
        Runs current timestep and saves results depending on chosen options.

        Returns
        -------
        timestep as month since 01-01-startyear, list with values of points of interest

        """
        month = self.taskdata['month']
        year = self.taskdata['year']
        result = self.run_ts(month, year)

        if isinstance(self.dconfig.pois, pd.DataFrame):
            # Get downscaled discharge time value for points of interest
            ts_values = []
            for indexp, prow in self.dconfig.pois.iterrows():
                ts_values.append(result[prow['row'], prow['col']])
        else:
            ts_values = []

        # Write downscaled discharge to disk
        if self.dconfig.write_result == 'raster':
            out_raster = os.path.join(self.dconfig.write_dir,
                                      '15s_dis_{}_{:02d}'.format(year, month))
            DownScaleArray(self.dconfig,
                           self.dconfig.aoi,
                           write_raster_trigger=True).load_data(result.astype(np.float32),
                                                                out_raster
                                                                )

        elif self.dconfig.write_result == 'nc': #netCDF
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
            ds.to_netcdf(
                os.path.join(
                    self.dconfig.write_dir,
                    '15s_dis_{}_{:02d}.nc4'.format(year, month)
                ),
                encoding={'dis': {'zlib': True, 'complevel': 9,
                                  'dtype': 'float32'}},
                unlimited_dims=['time']
            )

            del ds
            del result
        # self.wg.aoi = self.kwargs['area_of_interest']
        return month - 1 + (year - self.dconfig.startyear) * 12, ts_values

    def get_runoff_based_celldis(self, reliable_surfacerunoff_ar,
                             **kwargs):
        """
        Convert WG runoff at 30 min to net discharge in 15 sec cells, correcting for global lakes and reservoirs

        Parameters
        ----------
        reliable_surfacerunoff_ar: np.array
            lr array with reliable runoff with which the discharge is calculated
        kwargs: dict
            keyword arguments which are handed over to convert runoff to dis

        Previously two other arguments: lr_dis_series, netdis_30min_series
        But these arguments were not used in the end
        
        Returns
        -------

        """

        #Interpolate runoff from 30 min to 6 min with inverse-distance weighting
        if self.dconfig.sr_smoothing:
            # Remove outliers (2 times max of neighbours or half of min of neighbours)
            outlier_removed = self.remove_outliers_scipy(reliable_surfacerunoff_ar)
            tmp_ds = self.create_inmemory_30min_pointds(outlier_removed, all=True)
        else:
            tmp_ds = self.create_inmemory_30min_pointds(reliable_surfacerunoff_ar, all=True)
        tmp_interp = self.interpolation_to_grid(tmp_ds, '6min')
        del tmp_ds
        tmp_interp[tmp_interp == -99] = np.nan

        #if sr.smoothing == True -- Perform 5x5 mean filtering on 6 min raster (excluding NAs)
        if self.dconfig.sr_smoothing:
            tmp_smooth = self.spatial_window_averaging(tmp_interp, 5)
        else:
            tmp_smooth = tmp_interp

        #Disaggregate from 6 min to 15 s
        interpolated_smooth_15s = self.disaggregate(tmp_smooth, factor=24)
        del tmp_interp

        if self.dconfig.l12harm:
            # Correct for l12 catchment — not implemented
            masked_diss_tmp_smooth = self.harmonize_l12_hydrosheds(interpolated_smooth_15s)
        else:
            #Remove 15-sec cells where original HydroSHEDS pixel area raster is NoData
            masked_diss_tmp_smooth = self.mask_wg_with_hydrosheds(interpolated_smooth_15s)
        del interpolated_smooth_15s

        #Compute runoff-based discharge in the cell (i.e. convert runoff from mm to m3/s)
        conv = self.convert_runoff_to_dis(masked_diss_tmp_smooth, **kwargs)

        #Correct runoff-based discharge for changes in storage of global lakes and reservoirs
        # (i.e., redistribute volume changes from global water bodies' pour points to all cells intersecting with 
        # that global lake)
        if self.dconfig.correct_global_lakes:
            globallakes_fraction_15s = self.staticdata['globallakes_fraction_15s']

            #Disaggregate global lakes redistribution values from 30 min pd.Series in km3/y to 15 arc arrays in m3/s
            globallakes_addition_ar_15s_m3s = self.disaggregate_ar(
                self.get_30min_array(
                    self.taskdata['globallakes_addition']
                    / (self.daysinmonth_dict[kwargs['month']] * 24 * 60 * 60) * 1000000000,
                    float(0)),
                factor=120)

            #Redistribute storage change based on fraction of lakes in LR cell intersecting with HR cell
            conv = conv - (globallakes_fraction_15s * globallakes_addition_ar_15s_m3s)

            #Convert negative runoff-based discharge values to 0
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                conv = np.where(conv < 0, 0, conv)

        return conv

    def get_smoothed_runoff(self, runoff):
        """
        This method takes in :term:`lr` runoff (may be :term:`sr` or :term:`sr` + :term:`gwr`) and first drops grid
        cells where the land area fraction is below 0.5 %. Then those dropped grid cells and also grid cells which had
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

        mean_land_fraction = self.staticdata['mean_land_fraction']

        # Identify and remove grid cells with less than 5% land area
        not_reliable_arcids = mean_land_fraction[mean_land_fraction <= 0.5].index
        reliable_surface_runoff = (runoff.drop(not_reliable_arcids))

        #Convert pd series of surface runoff (with removed cells) to 30 min array
        upper = self.get_30min_array(reliable_surface_runoff, np.nan)
        #Run interpolation on a point-equivalent of the surface runoff raster
        points = self.create_inmemory_30min_pointds(upper)
        lower = self.interpolation_to_grid(ds=points,
                                           resolution='30min',
                                           alg="invdistnn:power=1.0:smoothing=0.0:radius=20"
                                               ":min_points=1:max_points=9:nodata=-99")
        #Fill dropped cells with interpolated values
        new_surface_runoff_land_mm = self.stack(upper, lower)

        return new_surface_runoff_land_mm

    def get_30min_array(self, s, nan=-99):
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
        wg_input = self.staticdata['wg_input']
        aoi = self.dconfig.aoi
        if isinstance(s, pd.Series):
            s.name = 'variable'
            df = wg_input.merge(s, left_index=True, right_index=True) #Append basic information
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

    def create_inmemory_30min_pointds(self, inp, **kwargs):
        """
        Method which creates an inmemory point layer from a WaterGAP resolution array (720*360) or
        a pandas Dataframe with arcid

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
                    x = aoi[0][0] + (idx[1] / 2) + 0.25 #Create point in the middle of cells (0.25 arc-degs from the edge)
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
            else:
                for x in coords.itertuples():
                    col = int((x.X - aoi[0][0]) // 0.5)
                    row = int((aoi[1][1] - x.Y) // 0.5)
                    if not np.isnan(inp[row, col]):
                        feat = ogr.Feature(lyr.GetLayerDefn())
                        # irow, icol = self.wg_input.data.loc[x.Index, ['GR.UNF2', 'GC.UNF2']]
                        feat.SetField("variable", inp[row, col])
                        pt = ogr.Geometry(ogr.wkbPoint)
                        pt.SetPoint(0, x.X, x.Y)
                        feat.SetGeometry(pt)
                        lyr.CreateFeature(feat)
                        feat.Destroy()
        return ds

    def interpolation_to_grid(self, ds, resolution, **kwargs):
        """

        Parameters
        ----------
        ds: data points
        resolution: desired resolution, accepts "6min", "30min", and "15s"
        kwargs: can provide "alg" argument to gdal.GridOptions

        Returns
        -------
        inmemory np array
        """

        aoi = self.dconfig.aoi

        #Determine number of rows and columns depending on desired output resolution
        if resolution == '6min':
            width = (aoi[0][1] - aoi[0][0]) * 10
            height = (aoi[1][1] - aoi[1][0]) * 10
        elif resolution == '30min':
            width = (aoi[0][1] - aoi[0][0]) * 2
            height = (aoi[1][1] - aoi[1][0]) * 2
        elif resolution == '15s':
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
        #Perform interpolation
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
        #Create output grid
        outr = gdal.Grid('outr', ds, options=go)
        res = outr.ReadAsArray().copy()
        del outr
        #Return interpolated grid inmemory
        return res

    def mask_wg_with_hydrosheds(self, wg):
        """
        Mask disaggregated a DownScaleArray with a Hydrosheds raster file like flow directions.

        This process masks the 15s from WaterGAP originating data (DownScaleArray) with Hydrosheds data. They must not
        be the same size but DownScaleArray must be at least the size of Hydrosheds array resp. raster. The
        DownScaleArray is then clipped to Hydrosheds array and grid cells which are nan in Hydrosheds array are set nan
        in DownScaleArray.

        :param wg: DownScale Array with 15s data originating from WaterGAP data
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
        #Get/run flow accumulation
        newar = self.staticdata['flowacc'].get(dis15s)
        newar[np.isnan(self.staticdata['pixelarea'])] = np.nan
        return newar

    def calculate_lr_correctionvalues(self, runoffbased_celldis_15s_ar, netdis_30min_series, **kwargs):
        """
        Compute correction values to route downstream
        The sum of disaggregated runoff-based discharge from all HR cells within each LR cell is compared to the net
        discharge which is generated in this LR cell (this serves to account for additional information from the
        routing routine of the LR GHM, in particular about the impact of surface water bodies and human water use on
        streamflow).

        Parameters
        ----------
        runoffbased_celldis_15s_ar
        netdis_30min_series
        kwargs

        Returns
        -------

        """

        # aggregate-sum 15-sec runoff-based cell discharge to 30-min runoff-based cell discharge (m3/s)
        runoffbased_celldis_30min_ar = self.aggsum(runoffbased_celldis_15s_ar, 120, zeroremove=False)
        
        #Convert 30-min net runoff to m3/s (from in km3/month if self.dconfig.mode == 'ts', else from km3/yr)
        if self.dconfig.mode == 'ts':
            netdis_30min_m3s_ar = (self.get_30min_array(netdis_30min_series, np.nan)
                              / (self.daysinmonth_dict[kwargs['month']] * 24 * 60 * 60) * 1000000000)
        else:
            netdis_30min_m3s_ar = self.get_30min_array(netdis_30min_series, np.nan) / (365 * 24 * 60 * 60) * 1000000000

        #Control for proportion of actual land (vs sea) in 30-min cell
        landratio_corr = self.staticdata['landratio_corr']
        netdis_30min_m3s_ar *= landratio_corr

        #Fill NAs in net runoff array with values from aggregated runoff-based cell discharge
        netdis_30min_m3s_ar = self.stack(upper=netdis_30min_m3s_ar, lower=runoffbased_celldis_30min_ar)

        #Compute correction value stemming from each cell (to be routed downstream)
        dif_dis_30min = netdis_30min_m3s_ar - runoffbased_celldis_30min_ar
        return dif_dis_30min

    def get_corrweight(self, runoffbased_celldis_15s_ar, weightingfactor=1):
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
        #Compute actual flow-accumulated discharge
        runoffbased_dis_15s = self.flow_acc(runoffbased_celldis_15s_ar)

        #Compute sum of discharge from all HR cells in each LR cell
        dis_fg_sum_30min = self.aggsum(runoffbased_dis_15s, 120, zeroremove=False)

        # Replace 0s with 1 to avoid dividing by 0, but won't influence outcome (dividing 0 by 1)
        dis_fg_sum_30min[dis_fg_sum_30min == 0] = 1

        # Compute proportion of discharge from LR cell contained in each HR cell
        dis_corr_weight = runoffbased_dis_15s / self.disaggregate_ar(dis_fg_sum_30min, 120)

        # To assign even more weight to cells with higher discharge.
        # Multiply the correction weight (above what it would be if discharge was evenly distributed) by a factor
        if weightingfactor != 1:
            #Compute
            dis_corr_weight = (
                    ((dis_corr_weight - (1/(120*120)))
                     * weightingfactor)
                    + (1/(120*120))
            )
        return dis_corr_weight

    def correct_dis(self, correction_grid_30min, correction_weights_15s, runoffbased_celldis_15s_ar,
                           correction_threshold=0.001, **kwargs):
        """
        Based on the raw discharge calculated with disaggregated runoff, a lr correction grid and correction weights
        the downscaled discharge is calculated.

        The correction grid is disaggregated to hr and multiplied with correction weights. Then the correction_threshold value is
        applied if parameter apply_thresh == True. This correction_threshold is a multiplicator for upstream area (i.e., m3/s/km2),
        which is then used as maximum correction value in positive or negative direction. Then the weighted and optional
        limited correction values are applied on the raw discharge calculated with disaggregated runoff. To avoid
        implausible artifacts of the correction. Negative discharge values are set to zero.

        Parameters
        ----------
        correction_grid_30min: np.array
            lr grid with correction values
        correction_weights_15s: np.array
            hr grid with weights for the lr correction values
        runoffbased_celldis_15s_ar: np.array
            grid with runoffbased_celldis_15s_ar
        correction_threshold: float or None
            multiplicator for upstream area to limit hr correction values, None if no correction_threshold should be applied

        Returns
        -------
        np.array
            hr grid with corrected discharge

        """
        #Apply correction weights to correction grid
        celldis_correctionvalue_15s = self.disaggregate_ar(correction_grid_30min.data, 120) * correction_weights_15s.data

        if correction_threshold is not None:
            ctar = self.staticdata['upstream_pixelarea'] * correction_threshold
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                # Create a grid to know which cells have negative correction
                corr_sign = np.where(celldis_correctionvalue_15s < 0, -1, 1)
            # Make sure the absolute value of the correction weights remain under the correction_threshold
            celldis_correctionvalue_15s = np.minimum(np.abs(celldis_correctionvalue_15s), ctar) 
            #Re-assign correct sign to each correction weight
            celldis_correctionvalue_15s = celldis_correctionvalue_15s * corr_sign
            del corr_sign
        
        #Apply correction values to flow-accumulated runoff-based discharge
        corrected_celldis = runoffbased_celldis_15s_ar + celldis_correctionvalue_15s
        
        #Accumulate corrected cell discharge to yield actual dischare
        corrected_dis = self.flow_acc(corrected_celldis)

        # manual correction of negative discharge
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            corrected_dis[corrected_dis < 0] = 0
            
        return corrected_dis

    def get_largerivers_correction_grid(self, wgdis_series_30min, precorrected_dis, correction_grid_30min, **kwargs):
        """
        This method adapts the correction grid to account for differences in river networks (i.e. missing endorheic
        sinks). This adaption is only done in large rivers ( geq 50000 km2)

        Parameters
        ----------
        dis
        precorrected_dis
        correction_grid_30min
        kwargs

        Returns
        -------

        """

        flowdir = self.get_30min_array('flowdir')
        largerivers_mask_30min = self.staticdata['largerivers_mask']
        
        #Convert discharge to m3/s (from km3/month if mode==ts or km3/yr if mode==long-term average
        if self.dconfig.mode == 'ts':
            wgdis_30min_m3s = self.get_30min_array(wgdis_series_30min, np.nan) / (self.daysinmonth_dict[kwargs['month']] * 24 * 60
                                                          * 60) * 1000000000
        else:
            wgdis_30min_m3s = self.get_30min_array(wgdis_series_30min, np.nan) / (365 * 24 * 60 * 60) * 1000000000
        
        #Compute net discharge in large river 30 min cells from WG discharge data
        wgdis_largerivers_30min = wgdis_30min_m3s * largerivers_mask_30min
        net_wgdis_largerivers_30min = (wgdis_largerivers_30min
                                    - get_inflow_sum(in_valuegrid=wgdis_largerivers_30min,
                                                     in_flowdir=flowdir)
                                    )
        #Compute maximum 15 sec accumulated pre-corrected discharge in each large river 30 min cell
        cell_pourpixel = self.staticdata['cell_pourpixel']
        precorrected_dis = precorrected_dis * cell_pourpixel
        accumulated_precorrecteddis_30min = self.aggmax(precorrected_dis, 120)
        accumulated_precorrecteddis_30min_largerivers = accumulated_precorrecteddis_30min * largerivers_mask_30min
        
        #Compute net pre-corrected discharge in each large river 30 min cell
        net_precorrecteddis_largerivers_30min = (accumulated_precorrecteddis_30min_largerivers
                                           - get_inflow_sum(in_valuegrid=accumulated_precorrecteddis_30min_largerivers,
                                                            in_flowdir=flowdir)
                                           )

        #Compute difference in net discharge from WG and from the accumulated pre-corrected discharge
        net_dis_dif_largerivers_30min = net_wgdis_largerivers_30min - net_precorrecteddis_largerivers_30min

        #Transfer net_dis_dif from outside the large-rivers mask to large rivers downstream
        gapmask_30min = 1 - largerivers_mask_30min
        transfer_value_grid = gapmask_30min * net_dis_dif_largerivers_30min #Isn't this all equal to 0? I think that net_dis_dif should be first computed outside of large rivers masks

        gap_flowdir_30min = flowdir * (1 - largerivers_mask_30min)
        gap_flowdir_30min[flowdir == -99] = -99

        gap_flowacc = self.staticdata['30mingap_flowacc']
        transfer_accu_grid = FlowAccTT(in_flowdir=gap_flowdir_30min,
                                       in_static_flowacc=gap_flowacc,
                                       pad=True).get(
            in_valuegrid=transfer_value_grid,
            no_negative_accumulation=False)

        #Compute final net discharge correction value for large rivers
        new_diff_dis_30min = correction_grid_30min + ((net_dis_dif_largerivers_30min + transfer_accu_grid)
                                                * largerivers_mask_30min)

        return new_diff_dis_30min

    def shift_correction_grid(self, correction_grid_30min):
        #For each cell, partially shift the correction value to the next LR downstream cell
        # if the maximum HR upstream area in the next LR downstream cell is at least 0.9*maximum HR upstream area in the cell
        # This process is to focus correction even more strongly on large rivers
        flowdir30min = self.get_30min_array('flowdir')
        corr_grid = ((correction_grid_30min * self.staticdata['keepgrid'])
                     + get_inflow_sum(in_valuegrid=(correction_grid_30min * self.staticdata['shiftgrid']),
                                      in_flowdir=flowdir30min)
                     )
        return corr_grid

    def smooth_correction_grid(self, correction_grid_30min):
        """
        For all cells for which the sign of the correction value does not match that of the downstream cell.
        Decrease the difference in values between the two cells by:
         - if the correction value in the upstream cell is negative and that of the downstream cell is negative,
            adding the minimum value among the two cells to the upstream cell
         - if the correction value in the upstream cell is positive and that of the downstream cell is positive,
            substract the minimum value among the two cells from the upstream cell
        Then balance the total correction by adding (substracting) to the downstream cell the sum of all values that
        were substracted (added) from upstream cells.
        -> Not fully sure that it's performing as desired. TBD

        Parameters
        ----------
        correction_grid_30min - grid of correction values to smooth

        Returns
        -------

        """

        flowdir30min = self.get_30min_array('flowdir')
        corr_grid = correction_grid_30min

        for i in range(10):
            downstream_corr_grid = get_downstream_grid(in_valuegrid=corr_grid, in_flowdir=flowdir30min, out_grid=None)
            #Identify the minimum absolute correction value between each cell and its downstream cell
            min_diff_grid = np.min([np.abs(corr_grid), np.abs(downstream_corr_grid)],
                                   axis=0)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                #Assign 1 where the correction value for a cell is negative and it is positive for the cell downstream
                #Assign -1 where the correction value for a cell is positive and it is negative for the cell downstream
                #0 when the correction value is the same sign in both cells
                sign_grid = (((corr_grid < 0) * (downstream_corr_grid > 0)).astype(int) -
                             ((corr_grid > 0) * (downstream_corr_grid < 0)).astype(int))
            #Change the sign of the upstream grid cell to match that of the downstream grid cell.
            #Don't apply a correction if they have the same sign
            change_grid = sign_grid * min_diff_grid

            #Balance change in upstream cells with the corresponding total change with the opposite sign in the
            #downstream cell to make sure that the total correction remains equal across the basin
            inflow_change_grid = get_inflow_sum(in_valuegrid=change_grid, in_flowdir=flowdir30min)
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
            division_term = (self.daysinmonth_dict[kwargs['month']] * 24 * 60 * 60)
        elif self.dconfig.mode == 'longterm_avg':
            division_term = (365 * 24 * 60 * 60)
        else:
            raise Exception()
        area = self.staticdata['pixelarea']
        data = (runoff * area * 1000 / division_term)
        return data

    @staticmethod
    def remove_outliers_scipy(ar):
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

        ar = generic_filter(input=ar, function=nanmean, size=windowsize)
        return ar

    @staticmethod
    def disaggregate(dsarray, factor):
        """
        Disaggregation of an array into higher resolution

        Parameters
        ----------
        dsarray : DownscalingArray
            array which should be disaggregated into higher resolution
        factor : number
            disaggregation factor (ratio of target resolution to original resolution) as integer
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

def run_task(task, path):
    dd = DryverDownscaling(in_taskdata_dict_picklepath=task,
                           in_staticdata_dict_picklepath= os.path.join(path, 'staticdata.pickle'),
                           in_config_dict_picklepath=os.path.join(path, 'config.pickle')
                           )
    return dd.save_and_run_ts()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run downscaling for one TS')
    parser.add_argument('Taskpart', metavar='T', type=int, help="which part should be worked on")
    parser.add_argument('path', metavar='path', type=str, help="the path where where the downscaling should happen")
    parser.add_argument('taskn', type=int, help="in how many groups the subtasks should be distributed")
    args = parser.parse_args()

    run_tasks = partial(run_task, path=args.path)
    tasklist = [task for task in glob.glob(os.path.join(args.path,'*task*.pickle'))]
    splitlist = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)]
    splitfactor = (-(-len(tasklist) // args.taskn))
    tasklist = splitlist(tasklist, splitfactor)
    poi_list = [run_tasks(x) for x in tasklist[args.Taskpart-1]]

    with open('{}result_part{:02d}.pickle'.format(args.path, args.Taskpart), 'wb') as f:
        pickle.dump(poi_list, f)
