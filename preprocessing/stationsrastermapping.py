import gdal
import pandas as pd
from osgeo import ogr


def create_stationrastermapping(flowaccpath, stations_vector,  outputdir):
    driver = ogr.GetDriverByName('GeoJSON')
    datasource = driver.Open(stations_vector)
    stations = datasource.GetLayer()
    gt = gdal.Open(flowaccpath).GetGeoTransform()

    def get_row_col(x, y):
        ulX = gt[0]
        ulY = gt[3]
        xDist = gt[1]
        col = int((x - ulX) / xDist)
        row = int((ulY - y) / xDist)
        return row, col

    dflist = []
    for station in stations:
        srow, scol = get_row_col(station.GetGeometryRef().GetX(), station.GetGeometryRef().GetY())
        dflist.append([station.GetField('dd_id'), srow, scol, station.GetField('source')])
    df = pd.DataFrame(dflist)
    df.columns = ['stationid', 'row', 'col', 'metasource']
    df.to_csv('{}stations.csv'.format(outputdir))


if __name__ == '__main__':
    create_stationrastermapping('/home/home8/dryver/hs_reproduced/euassi_dir_15s.tif', '/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/04_stations/europe_stations.geojson', '/home/home8/dryver/hs_reproduced/')