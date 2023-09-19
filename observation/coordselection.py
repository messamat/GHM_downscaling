import osgeo.gdal
import pandas as pd
import numpy as np


def main():
    fa_path = '/home/home8/dryver/hydrosheds/rhone_flowacc_15s.tif'

    calstations = pd.read_csv('calstations.csv', sep=';', decimal=',')
    fa = gdal.Open(fa_path)
    faar = fa.ReadAsArray()
    leftuppercoord = round(fa.GetGeoTransform()[0]), round(fa.GetGeoTransform()[3])
    xwgcells = faar.shape[1]//120
    ywgcells = faar.shape[0]//120
    xmin, xmax, ymax, ymin = (leftuppercoord[0], leftuppercoord[0] + 0.5 * xwgcells,
                              leftuppercoord[1], leftuppercoord[1] - 0.5 * ywgcells)
    arrayindex = {}
    arrayindex2 = {}
    for stix in range(calstations.shape[0]):
        lat = calstations.loc[stix, 'lat_ddm']
        lon = calstations.loc[stix, 'lon_ddm']
        if xmin < lon < xmax and ymin < lat < ymax:
            wgcolindex = (lon - xmin)//0.5
            wgrowindex = (ymax - lat) // 0.5
            hydrosheds_colindex = int(wgcolindex * 120)
            hydrosheds_rowindex = int(wgrowindex * 120)
            sel = faar[hydrosheds_rowindex:hydrosheds_rowindex + 120, hydrosheds_colindex: hydrosheds_colindex+120]
            if np.max(sel) > -99:
                ind = np.unravel_index(np.argmax(sel, axis=None), sel.shape)
                arrayindex[calstations.loc[stix, 'grdc_no']] = {'idx': (ind[0] + hydrosheds_rowindex, ind[1] + hydrosheds_colindex),
                                                                'meta': calstations.loc[stix, :]}
                arrayindex2[str(calstations.loc[stix, 'grdc_no'])] = [ind[0] + hydrosheds_rowindex, ind[1] + hydrosheds_colindex,
                                                                 calstations.loc[stix, 'to_ArcIDWL'],
                                                                 'calstation']
    # mm stations
    mm = gdal.Open('/home/home8/dryver/observed_messager/forDRYVER20210316/mm_stations_PointToRaster_rhoneextended.tif')
    mmmeta = pd.read_csv('/home/home8/dryver/observed_messager/forDRYVER20210316/stations_meta_210324_witharcid.csv', sep=';')
    mmar = mm.ReadAsArray()
    mmar[mmar == mm.GetRasterBand(1).GetNoDataValue()] = 0
    stationix = np.where(mmar > 0)
    for stn in range(len(stationix[0])):
        row = stationix[0][stn]
        col = stationix[1][stn]
        stationid = mmmeta.iloc[mmar[row, col], 2]
        if stationid in arrayindex2:
            arcid = arrayindex2[stationid][2]
            origin = 'both'
        else:
            arcid = mmmeta.iloc[mmar[row, col], -1]
            origin = 'mmprocessedstation'
        arrayindex2[mmmeta.iloc[mmar[row, col], 2]] = [row, col, arcid, origin]

    df = pd.DataFrame(arrayindex2).T.reset_index()
    df.columns = ['stationid', 'row', 'col', 'arcid', 'origin']
    df.to_csv('grdc_hydrosheds_rowcol_extended_rhone.csv')
    pass


if __name__ == '__main__':
    main()
