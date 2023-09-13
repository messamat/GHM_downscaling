
import gdal
import numpy as np
from codetiming import Timer

from ModifiedFlowAcc import FlowAccTT
"""
HSspecific data 
_lev12_15s.tif
_dir_geq0_15s.tif
_flowacc_15s.tif
_cellpourpoint_15s.tif
shiftPercentGridv9.tif
largeRiverMask.tif
_pixarea_15s.tif
"""


class HSData:
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
        self.fd = self.read_flowdir()
        self.hsgt = self.fd.GetGeoTransform()
        self.flowacc = FlowAccTT(self.fd.ReadAsArray(), self.get_flowacc_path())
        self.paar, self.uparea = self.get_pixarea_upstream_area()
        self.glolakresfrar = self.get_glolakresfr()
        self.keepGrid, self.shiftGrid = self.downstream_shift_grids()
        if config.l12harm:
            self.l12fp = self.config.hs_path + self.config.continent + '_lev12_15s.tif'
            self.l12harmdata = self.prepare_level12_harmonize()

    def read_flowdir(self):
        """
        Reads in flow directions into gdal.DataSet

        Returns
        -------
        gdal.DataSet
        """
        flowdir = self.config.hs_path + self.config.continent + '_dir_15s.tif'
        fd = gdal.Open(flowdir)
        return fd

    def get_flowacc_path(self):
        return self.config.hs_path + self.config.continent + '_acc_15s.tif'

    def get_cellpourpixel(self):
        pp = self.config.hs_path + self.config.continent + '_cellpourpoint_15s.tif'
        pp = gdal.Open(pp)
        ar = pp.ReadAsArray()
        nav = pp.GetRasterBand(1).GetNoDataValue()
        ar[ar == nav] = 0
        return ar

    def get_pixarea_upstream_area(self):
        # pixelarea
        pixelareapath = self.config.hs_path + self.config.continent + '_pixarea_15s.tif'
        pa = gdal.Open(pixelareapath)
        paar = pa.ReadAsArray().copy()
        paar[paar == pa.GetRasterBand(1).GetNoDataValue()] = np.nan
        uparea = self.flowacc.get(paar)
        uparea[np.isnan(paar)] = np.nan
        del pa
        return paar, uparea

    def get_glolakresfr(self):
        glolakresfrpath = self.config.hs_path + self.config.continent + '_pixareafraction_glolakres_15s.tif'
        glolakresfr = gdal.Open(glolakresfrpath)
        glolakresfrar = glolakresfr.ReadAsArray().copy()
        glolakresfrar[glolakresfrar == glolakresfr.GetRasterBand(1).GetNoDataValue()] = 0
        return glolakresfrar

    def downstream_shift_grids(self):
        shiftGrid = '{}{}_shiftgrid.tif'.format(self.config.hs_path, self.config.continent)
        keepGrid = '{}{}_keepgrid.tif'.format(self.config.hs_path, self.config.continent)
        kg = self.get_wg_corespondend_grid(keepGrid)
        sg = self.get_wg_corespondend_grid(shiftGrid)
        return kg, sg

    def get_wg_corespondend_grid(self, fp):
        ra = gdal.Open(fp)
        xoff = abs((round(ra.GetGeoTransform()[0] - self.fd.GetGeoTransform()[0])) * 2)
        yoff = ((round(ra.GetGeoTransform()[3]) - round((self.fd.GetGeoTransform()[3]))) * 2)
        xsize = self.fd.RasterXSize // 120
        ysize = self.fd.RasterYSize // 120
        raar = ra.ReadAsArray(xoff=xoff, yoff=yoff, xsize=xsize, ysize=ysize)
        if raar.dtype.kind == 'f':
            raar[raar == ra.GetRasterBand(1).GetNoDataValue()] = np.nan
        return raar

    def largerivermask(self):
        lriver = '{}{}_largeRiverMask.tif'.format(self.config.hs_path, self.config.continent)
        return self.get_wg_corespondend_grid(lriver)

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
        pixareapath = self.config.hs_path + self.config.continent + '_pixarea_15s.tif'
        pa = gdal.Open(pixareapath)
        return pa
