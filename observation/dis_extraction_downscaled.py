import xarray as xr
import pandas as pd
import netCDF4
from dask import compute


def generate_df(path, timesteps, stationrowcol):
    files = []
    for ts in timesteps:
        fn = '15sec_dis_{}_{:02d}.nc4'.format(ts[1], ts[0])
        files.append(path+fn)
    ds = xr.open_mfdataset(files, decode_times=False, parallel=True)
    dfb = {}
    for station in stationrowcol.iterrows():
        dfb[station[1]['grdcno']] = ds.dis[station[1]['row'], station[1]['col'], :].compute()
        print('one station done')
    df = pd.DataFrame.from_dict(dfb)
    df.to_pickle('downscaledtssel.pdpickle')
    df.to_csv('downscaledsel.csv')
    return 0

def generate_df2(path, timesteps, stationrowcol):
    files = []
    for ts in timesteps:
        fn = '15sec_dis_{}_{:02d}.nc4'.format(ts[1], ts[0])
        files.append(path+fn)
    dfb = {}
    for station in stationrowcol.iterrows():
        values = []
        for file in files:
            ncone = netCDF4.Dataset(file)
            values.append(ncone.variables['dis'][station[1]['row'], station[1]['col'], 0])
        dfb[station[1]['grdcno']] = values
        print('one station done')
    df = pd.DataFrame.from_dict(dfb)
    df.to_pickle('downscaledtssel2.pdpickle')
    df.to_csv('downscaledsel2.csv')
    return 0

if __name__ == '__main__':
    timesteps = []
    for yr in range(1981, 1990):
        for mon in range(1, 13):
            timesteps.append([mon, yr])
    stations = pd.read_csv('grdc_hsrowcol.csv')
    stations = stations[[x in (6139100, 6139390, 6742800, 6742900, 6842800, 6842900, 6939050, 6975110, 6977100) for x in stations.grdcno]]
    df = generate_df2('/home/home8/dryver/rastertemp/nobackup/', timesteps, stations)
