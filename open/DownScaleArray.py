from os import path

import pandas as pd
import numpy as np
from osgeo import gdal, osr


class DownScaleArray:
    """
    Array which holds data for the downscaling process.

    This array is aware of resolution of the containing data,
    it can write rasters with its data and can be pickled and
    read again

    Parameters
    ----------
    config : DownscalingConfig
        configuration object for running the downscaling,
        it is used to know if and where to write the data
    aoi :  ((number, number), (number, number))
        area of interest in degree aoi[0][0] = left border,
        aoi[0][1] = right border, aoi[1][0] = lower and aoi[1][1] = upper
    **kwargs : dict, optional
        keyword arguments

    Attributes
    ----------

    status : None or str
         initialized with None. Later holds information on resolution and name of the
         downscaled array in following manner *'resolution_name'* e.g. *'15s_dis'*
    data : None or numpy.array
         initialized with None. Later holds data as numpy.array
    write_raster_specs : dict
        dict containing information on how big is the conversion factor for one
        axis from one degree to the specified resolution (currently implemented:
        30min, 6min, 30sec, 15s)
    """
    def __init__(self, config, aoi, **kwargs):
        self.status = None
        self.data = None
        self.config = config
        self.aoi = aoi
        self.write_raster_trigger = config.write_raster

        if 'rasterdtype' in kwargs:
            gdaldtypdict = {'f32': gdal.GDT_Float32,
                            'f64': gdal.GDT_Float64}
            self.rasterdtype = gdaldtypdict[kwargs['rasterdtype']]
        else:
            self.rasterdtype = gdal.GDT_Float32

        self.write_raster_specs = {
            '30min': 2,
            '6min': 10,
            '30sec': 120,
            '15s': 240
        }

        if 'write_raster_trigger' in kwargs:
            self.write_raster_trigger = kwargs['write_raster_trigger']

    def write_raster(self, fn):
        """
        Writes a raster with data in data attribute.

        Uses the create_raster method to write the file.

        Parameters
        ----------
        fn : str
            path to where the raste rshould be written to

        Returns
        -------
        None
        """
        #conversion factor between one degree and the cell size (e.g., 2 for 30 min cells, 240 for 15 arc-sec)
        rmulti = self.write_raster_specs[self.status.split('_')[0]]
        #Get number of cols and rows based on extent and conversion factor
        no_cols = (self.aoi[0][1] - self.aoi[0][0]) * rmulti
        no_rows = (self.aoi[1][1] - self.aoi[1][0]) * rmulti
        cellsize = 1 / rmulti
        leftdown = (self.aoi[0][0], self.aoi[1][0])  #lower-left corner of extent
        grid_specs = (int(no_rows), int(no_cols), cellsize, leftdown)
        self.create_raster(self.data, grid_specs, to_file=fn, dtype=self.rasterdtype)

    def load_data(self, data, status):
        """
        Loads data and status into instance of DownScaleArray

        Parameters
        ----------
        data : np.array
            data array you want to store
        status : str
            name of data incorporating resolution information pattern: *'resultion_name'*

        Returns
        -------
        self
        """
        self.data = data
        self.status = status

        if self.write_raster_trigger:
            self.write_raster(path.join(self.config.temp_dir, 'r', self.status, '.tif'))
        return self

    def load(self, path_saved, status):
        """
        Load pickle from path, can be pdpickle(.pdpickle), numpy pickle (.npy), raster (.tif)

        Parameters
        ----------
        path_saved : str
            path were the pickle is saved
        status : str
            string in pattern *'resolution_name'*  see also Attributes of DownScaleArray

        Returns
        -------
        self
        """
        if path_saved.endswith('.pdpickle'):
            self.data = pd.read_pickle(path_saved)
        elif path_saved.endswith('.npy'):
            self.data = np.load(path_saved)
        elif path_saved.endswith('.tif'):
            self.data = gdal.Open(path_saved)
        else:
            Exception('Loading not implemented')
        self.status = status
        return self

    def save_pickle(self, name):
        """
        Saves the data in the attribute data in a pickle either pandas and numpy pickle

        Takes name attribute to write the file to config.temp_dir.

        Parameters
        ----------
        name : str
             name of the pickle which you want to save

        Returns
        -------
        None
        """
        if isinstance(self.data, pd.DataFrame):
            self.data.to_pickle(path.join(self.config.temp_dir, name + '.pdpickle'))
        elif isinstance(self.data, np.ndarray):
            np.save(path.join(self.config.temp_dir, name), self.data)
        else:
            Exception('Saving not implemented')

    @staticmethod
    def create_raster(ar, grid_spec, in_epsg=4326, **kwargs):
        """
        Method to generate an inmemory raster or write out a raster to file based on an array.
        If it shall be written the parameter 'to_file' must be handed over.

        :param ar: numpy array which shall be transformed into raster
        :param grid_spec: specification of rastergrid (number of rows, number of cols, size of gridcell,
         (x,y) of lowerleft gridorigin
        :param kwargs:
        :return: returns 0 if raster is written to file otherwise returns inmemory raster
        """
        dtype = gdal.GDT_Float32

        if 'dtype' in kwargs:
            dtype = kwargs['dtype']

        rows, cols, size, origin = grid_spec
        originx, originy = origin

        if 'to_file' in kwargs:
            driver = gdal.GetDriverByName('GTiff')
            name = kwargs['to_file']
        else:
            driver = gdal.GetDriverByName('MEM')
            name = 'r'

        out_raster = driver.Create(name, cols, rows, 1, dtype)
        out_raster.SetGeoTransform((originx, size, 0, originy, 0, size))
        out_band = out_raster.GetRasterBand(1)
        out_band.WriteArray(ar[::-1])
        out_band.SetNoDataValue(-99.)
        out_raster_srs = osr.SpatialReference()
        out_raster_srs.ImportFromEPSG(in_epsg)
        out_raster.SetProjection(out_raster_srs.ExportToWkt())

        if 'to_file' in kwargs:
            out_band.FlushCache()
            out_raster.FlushCache()
            outband = None
            out_raster = None
            return 0
        else:
            return out_raster
