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
        fdband = fd.GetRasterBand(1)
        fdar = fdband.ReadAsArray()
        grar = gridband.ReadAsArray()

    newar = grar.copy()
    for dire, xm, ym in [(1, 0, 1), (2, 1, 1), (4, 1, 0), (8, 1, -1), (16, 0, -1),
                         (32, -1, -1), (64, -1, 0), (128, -1, 1)]:
        ix = np.where(fdar == dire)
        downxix = ix[0] + xm
        downyix = ix[1] + ym
        downxix[downxix == grar.shape[0]] = grar.shape[0]-1
        downyix[downyix == grar.shape[1]] = grar.shape[1]-1
        downxix[downxix < 0] = 0
        downyix[downyix < 0] = 0
        newar[ix] = grar[downxix, downyix]
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
