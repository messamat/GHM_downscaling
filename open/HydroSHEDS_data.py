
import gdal
import numpy as np
from codetiming import Timer

from open.ModifiedFlowAcc import FlowAccTT
"""
HydroSHEDSspecific data 
_lev12_15s.tif
_dir_geq0_15s.tif
_flowacc_15s.tif
_cellpourpoint_15s.tif
shiftPercentGridv9.tif
largeRiverMask.tif
_pixarea_15s.tif
"""


class HydroSHEDS_data:
    """
    HydroSHEDS data read in and processed for Downscaling

    Parameters
    ----------
    config : DownscalingConfig
        Downscaling configuration object
    **kwargs : dict, optional
        keyword arguments
    """
    def __init__(self, config, **kwargs):
        self.config = config
        if 'aoi' in kwargs:
            self.aoi = kwargs['aoi']
        self.flowdir = self.read_flowdir()
        self.hydrosheds_geotrans = self.flowdir.GetGeoTransform()
        self.flowacc = FlowAccTT(self.flowdir.ReadAsArray(), self.get_flowacc_path())
        self.pixarea, self.uparea = self.get_pixarea_upstream_area()
        self.globallakes_fraction_ar = self.get_globallakes_fraction()
        self.keepGrid, self.shiftGrid = self.downstream_shift_grids()
        if config.l12harm:
            self.l12fp = self.config.hydrosheds_path + self.config.continent + '_lev12_15s.tif'
            self.l12harmdata = self.prepare_level12_harmonize()

    def read_flowdir(self):
        """
        Reads in flow directions into gdal.DataSet

        Returns
        -------
        gdal.DataSet
        """
        flowdir = self.config.hydrosheds_path + self.config.continent + '_dir_15s.tif'
        flowdir = gdal.Open(flowdir)
        return flowdir

    def get_flowacc_path(self):
        return self.config.hydrosheds_path + self.config.continent + '_acc_15s.tif'

    def get_cellpourpixel(self):
        pp = self.config.hydrosheds_path + self.config.continent + '_cellpourpoint_15s.tif'
        pp = gdal.Open(pp)
        ar = pp.ReadAsArray()
        nav = pp.GetRasterBand(1).GetNoDataValue()
        ar[ar == nav] = 0
        return ar

    def get_pixarea_upstream_area(self):
        # pixelarea
        pixelarea_path = self.config.hydrosheds_path + self.config.continent + '_pixarea_15s.tif'
        pa = gdal.Open(pixelarea_path)
        pixarea = pa.ReadAsArray().copy()
        pixarea[pixarea == pa.GetRasterBand(1).GetNoDataValue()] = np.nan
        uparea = self.flowacc.get(pixarea)
        uparea[np.isnan(pixarea)] = np.nan
        del pa
        return pixarea, uparea

    def get_globallakes_fraction(self):
        globallakes_fraction_path = self.config.hydrosheds_path + self.config.continent + '_pixareafraction_glolakres_15s.tif'
        globallakes_fraction = gdal.Open(globallakes_fraction_path)
        globallakes_fraction_ar = globallakes_fraction.ReadAsArray().copy()
        globallakes_fraction_ar[globallakes_fraction_ar == globallakes_fraction.GetRasterBand(1).GetNoDataValue()] = 0
        return globallakes_fraction_ar

    def downstream_shift_grids(self):
        shiftGrid = '{}{}_shiftgrid.tif'.format(self.config.hydrosheds_path, self.config.continent)
        keepGrid = '{}{}_keepgrid.tif'.format(self.config.hydrosheds_path, self.config.continent)
        kg = self.get_wg_corresponding_grid(keepGrid)
        sg = self.get_wg_corresponding_grid(shiftGrid)
        return kg, sg

    def get_wg_corresponding_grid(self, fp):
        ra = gdal.Open(fp)
        xoff = abs((round(ra.GetGeoTransform()[0] - self.flowdir.GetGeoTransform()[0])) * 2)
        yoff = ((round(ra.GetGeoTransform()[3]) - round((self.flowdir.GetGeoTransform()[3]))) * 2)
        xsize = self.flowdir.RasterXSize // 120
        ysize = self.flowdir.RasterYSize // 120
        raar = ra.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize)
        if raar.dtype.kind == 'f':
            raar[raar == ra.GetRasterBand(1).GetNoDataValue()] = np.nan
        return raar

    def largerivermask(self):
        lriver = '{}{}_largeRiverMask.tif'.format(self.config.hydrosheds_path, self.config.continent)
        return self.get_wg_corresponding_grid(lriver)

    @Timer(name='decorator', text='preparing the l12 harmonization took {seconds:.0f}s')
    def prepare_level12_harmonize(self):
        l12 = gdal.Open(self.l12fp)
        navalue = l12.GetRasterBand(1).GetNoDataValue()
        l12ar = l12.ReadAsArray()
        l12arix = np.arange(l12ar.size, dtype=np.int32)
        l12flat = l12ar.flatten()
        del l12ar
        l12arix = l12arix[~(l12flat == navalue)]
        l12flat = l12flat[~(l12flat == navalue)]
        sortindex = np.argsort(l12flat).astype(np.int32)
        l12arix = l12arix[sortindex]
        l12flat = l12flat[sortindex]
        splitar = np.where(np.diff(l12flat))[0] + 1
        return np.split(l12arix, splitar)

    def read_pixarea(self):
        pixareapath = self.config.hydrosheds_path + self.config.continent + '_pixarea_15s.tif'
        pa = gdal.Open(pixareapath)
        return pa
