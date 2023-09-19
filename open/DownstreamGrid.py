import numpy as np
from osgeo import gdal


def get_downstream_grid(in_valuegrid, in_flowdir, out_grid):
    """
    Generate a new raster with the value of the next downstream cell from in_valuegrid.

    in_valuegrid and in_flowdir must have the same grid dimensions

    :param in_valuegrid: raster filename or raster array whose values to assign to upstream cells
    :type in_valuegrid: str or np.ndarray
    :param in_flowdir: flowdir rastername in arcgis format (possible values: 0, 1, 2, 4, 8, 16, 32, 64, 128)
    :param out_grid: out rastername
    :return: raster or np.ndarray
    """

    #Read value grid and flowdir grid, whether directly passed as a numpy array or as a raster path
    if isinstance(in_valuegrid, np.ndarray):
        mode = 'np'
        flowdir_ar = in_flowdir
        value_ar = in_valuegrid
        if flowdir_ar.shape != value_ar.shape:
            raise Exception("value grid and flowdir not same dimension")
    else:
        mode = 'raster'
        raster = gdal.Open(in_valuegrid)
        flowdir_ras = gdal.Open(in_flowdir)
        if raster.RasterXSize != flowdir_ras.RasterXSize or raster.RasterYSize != flowdir_ras.RasterYSize:
            raise Exception("value grid and flowdir not same dimension")
        gridband = raster.GetRasterBand(1)
        value_ar = gridband.ReadAsArray()

        flowdir_band = flowdir_ras.GetRasterBand(1)
        flowdir_ar = flowdir_band.ReadAsArray()


    newar = value_ar.copy()

    # List of tuples showing, for each value in the direction raster (first value in each tuple), what the upstream
    # direction translates to in terms of cell shift along the rows (second value in each tuple) and
    # columns (third value in each tuple) of the raster (the xm and ym DO NOT correspond to longitude and latitude but
    # row and column indices in numpy arrays).
    #The original flow direction grid values correspond to the following:
    #       | 32| 64|128|
    #       | 16|   |1  |
    #       |  8|  4|2  |
    # For example, (1, 0, 1) therefore, a flowdir of 1 means a shift by 0 rows (x) and 1 column (y)

    flowdir_conversion_list = [(1, 0, 1), (2, 1, 1),
                               (4, 1, 0), (8, 1, -1),
                               (16, 0, -1),(32, -1, -1),
                               (64, -1, 0), (128, -1, 1)]

    #For each flow direction value
    for dire, xm, ym in flowdir_conversion_list:
        ix = np.where(flowdir_ar == dire)
        #Get downstream x and y for all cells with that flow direction value
        downxix = ix[0] + xm
        downyix = ix[1] + ym
        #Make sure the downstream indices lie within the bounds of the value grid
        downxix[downxix == value_ar.shape[0]] = value_ar.shape[0]-1
        downyix[downyix == value_ar.shape[1]] = value_ar.shape[1]-1
        downxix[downxix < 0] = 0
        downyix[downyix < 0] = 0
        #Assign grid value of the downstream area to each pixel
        newar[ix] = value_ar[downxix, downyix]

    if mode == 'raster':  #Write out to a raster if mode==raster
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.CreateCopy(out_grid, raster, 0)
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(newar)
        outband.FlushCache()
        outRaster.FlushCache()
        outband = None
        outRaster = None
        raster = None
        fd = None
        return 0
    else:   #Otherwise, return inmemory
        tmpar = value_ar.copy()
        tmpar[~np.isnan(newar)] = newar[~np.isnan(newar)]
        return tmpar

def get_inflow_sum(in_valuegrid, in_flowdir):
    """
    For each cell, sum values in all cells immediately upstream (i.e., single cell downstream routing).

    Parameters
    ----------
    :param in_valuegrid: raster array whose values to route and sum
    :param in_flowdir: raster array of flow directions. should be the same dimensions as in_valuegrid

    :return: raster array of inflow sums
    -------

    """
    if in_valuegrid.shape != in_flowdir.shape:
        raise Exception("grid and flowdir not same dimension")
    flowdir_pad = np.pad(in_flowdir, 1, 'constant', constant_values=0)
    grid_pad = np.pad(in_valuegrid, 1, 'constant', constant_values=0)
    new_ar = np.zeros(grid_pad.shape)

    for dire, xm, ym in [(1, 0, 1), (2, 1, 1), (4, 1, 0), (8, 1, -1), (16, 0, -1),
                         (32, -1, -1), (64, -1, 0), (128, -1, 1)]:
        ix = np.where(flowdir_pad == dire)
        inflow_values = grid_pad[ix]
        inflow_values[np.isnan(inflow_values)] = 0
        new_ar[ix[0] + xm, ix[1] + ym] += inflow_values

    new_ar[np.isnan(grid_pad)] = np.nan

    return new_ar[1:-1, 1:-1]


def main():
    get_downstream_grid(in_valuegrid="D:/tmp/gen_shiftpercentgrid/MaxUpArea_30min.tif",
                        in_flowdir="D:/tmp/gen_shiftpercentgrid/flowdir_30min.tif",
                        out_grid="D:/tmp/gen_shiftpercentgrid/MaxUpArea_down_30min.tif")


if __name__ == '__main__':
    main()
