import numpy as np
from osgeo import gdal


def get_downstream_grid(grid, flowdir, outgrid):
    """
    Generate a new raster with the value of the next downstream cell.

    Inputgrid and flowdir have to be the same grid.

    :param grid: raster filename or raster array
    :type grid: str or np.ndarray
    :param flowdir: flowdir rastername
    :param outgrid: out rastername
    :return: raster or np.ndarray
    """
    if isinstance(grid, np.ndarray):
        mode = 'np'
        fdar = flowdir
        grar = grid
        if fdar.shape != grar.shape:
            raise Exception("grid and flowdir not same dimension")
    else:
        mode = 'raster'
        raster = gdal.Open(grid)
        fd = gdal.Open(flowdir)
        if raster.RasterXSize != fd.RasterXSize or raster.RasterYSize != fd.RasterYSize:
            raise Exception("grid and flowdir not same dimension")
        gridband = raster.GetRasterBand(1)
        grar = gridband.ReadAsArray()

        fdband = fd.GetRasterBand(1)
        fdar = fdband.ReadAsArray()


    newar = grar.copy()

    #List of tuples showing for each value in the direction raster (first value in each tuple), what the upstream
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

    for dire, xm, ym in flowdir_conversion_list:
        ix = np.where(fdar == dire)
        downxix = ix[0] + xm
        downyix = ix[1] + ym
        ##Tes what these lines really do
        downxix[downxix == grar.shape[0]] = grar.shape[0]-1 #For the last row, assign the values of the second to last row?
        downyix[downyix == grar.shape[1]] = grar.shape[1]-1 #For the last column, assign the values of the second to last column?
        ##
        downxix[downxix < 0] = 0
        downyix[downyix < 0] = 0
        newar[ix] = grar[downxix, downyix] #Assign grid value of the downstream area to each pixel

    if mode == 'raster':
        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.CreateCopy(outgrid, raster, 0)
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(newar)
        outband.FlushCache()
        outRaster.FlushCache()
        outband = None
        outRaster = None
        raster = None
        fd = None
        return 0
    else:
        tmpar = grar.copy()
        tmpar[~np.isnan(newar)] = newar[~np.isnan(newar)]
        return tmpar


def get_inflow_sum(grid, flowdir):
    if grid.shape != flowdir.shape:
        raise Exception("grid and flowdir not same dimension")
    flowdir = np.pad(flowdir, 1, 'constant', constant_values=0)
    grid1 = np.pad(grid, 1, 'constant', constant_values=0)
    newar = np.zeros(grid1.shape)
    for dire, xm, ym in [(1, 0, 1), (2, 1, 1), (4, 1, 0), (8, 1, -1), (16, 0, -1),
                         (32, -1, -1), (64, -1, 0), (128, -1, 1)]:
        ix = np.where(flowdir == dire)
        inflow_values = grid1[ix]
        inflow_values[np.isnan(inflow_values)] = 0
        newar[ix[0] + xm, ix[1] + ym] += inflow_values
    newar[np.isnan(grid1)] = np.nan
    return newar[1:-1, 1:-1]


def main():
    get_downstream_grid("D:/tmp/gen_shiftpercentgrid/MaxUpArea_30min.tif",
                        "D:/tmp/gen_shiftpercentgrid/flowdir_30min.tif",
                        "D:/tmp/gen_shiftpercentgrid/MaxUpArea_down_30min.tif")


if __name__ == '__main__':
    main()
