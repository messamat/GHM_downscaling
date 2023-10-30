import osgeo.gdal
import pandas as pd
from osgeo import ogr, gdal

def create_stationrastermapping(in_flowacc_path, in_stations_vector,  outputdir):
    driver = ogr.GetDriverByName('GeoJSON')
    datasource = driver.Open(in_stations_vector)
    stations = datasource.GetLayer()
    flowacc = gdal.Open(in_flowacc_path)
    gt = flowacc.GetGeoTransform()

    #Make sure that stations and flowacc have the same coordinate system
    flowacc_prj = flowacc.GetSpatialRef()
    stations_prj = stations.GetSpatialRef()


    if flowacc_prj.IsSame(stations_prj) == 1:
        def get_row_col(x, y):
            ulX = gt[0] #upper left corner x (longitude) in projection coordinates
            ulY = gt[3] #upper left corner y (latitude) in projection coordinates
            xDist = gt[1] #pixel width in the linear unit of the projection coordinate system
            col = int((x - ulX) / xDist)  #col index in raster
            row = int((ulY - y) / xDist) #row index in raster
            return row, col

        dflist = []
        for station in stations:
            srow, scol = get_row_col(station.GetGeometryRef().GetX(), station.GetGeometryRef().GetY())
            dflist.append([station.GetField('dd_id'), srow, scol, station.GetField('source')])
        df = pd.DataFrame(dflist)
        df.columns = ['stationid', 'row', 'col', 'metasource']
        df.to_csv('{}stations.csv'.format(outputdir))
    else:
        raise Exception("Coordinate system of station vector does not match that of flow accumulation raster")


if __name__ == '__main__':
    create_stationrastermapping('/home/home8/dryver/hs_reproduced/euassi_dir_15s.tif', '/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/04_stations/europe_stations.geojson', '/home/home8/dryver/hs_reproduced/')