from os import path
import glob

import pandas as pd
from pyproj import Transformer
from numpy import nan, nanmean

from observation.DailyStation import DailyStationData


class RBISStation(DailyStationData):

    def __init__(self, fpath, station_data):
        super().__init__(station_data)
        self.fpath = fpath

    def read_station_data(self):
        with open(self.fpath, mode='r', encoding='utf-8-sig') as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith('#'):
                    if line.__contains__(':'):
                        self.station_data[line.split(':')[0][1:]] = line.split(':')[1].replace('\n', '').replace('\t','')
                    else:
                        self.station_data['columns'] = line.replace('\n', '').replace('#', '').split('\t')
        if self.station_data['Spatial Reference System'] not in ('WGS 84 (lat/long)'):
            epsglookup = {
                'ETRS89 / ETRS-LAEA': 25830,
                'WGS 84 / UTM zone 33N': 32633,
                'WGS 84 / UTM zone 30N': 32630

            }
            transformer = Transformer.from_crs('epsg:{}'.format(
                                                   epsglookup[self.station_data['Spatial Reference System']]),
                                               'epsg:4326')
            # treat mixed coordinates

            lon, lat = transformer.transform(self.station_data['Longitude'],
                                             self.station_data['Latitude'])
            if lon > 30:
                self.station_data['Latitude'] = lon
                self.station_data['Longitude'] = lat
            else:
                self.station_data['Latitude'] = lat
                self.station_data['Longitude'] = lon

    def read_daily_data(self):
        if not path.exists(self.fpath):
            print('path doesnt exist')
            raise OSError
        else:
            self.read_station_data()

        df = pd.read_csv(self.fpath,
                         comment='#',
                         sep='\t',
                         parse_dates=[1],
                         engine='python',
                         header=None
                         )
        df.columns = self.station_data['columns']
        df['date'] = pd.to_datetime(df['Date'])
        df = df.set_index(['date'])
        s = df.loc[:, 'discharge (cbm/s)']
        s = s[s != -9999]
        if s.min() < 0:
            s = s[s >= 0]
        s = s.dropna()
        complete_ts = pd.date_range(start=s.index.min(), end=s.index.max(), freq='D')
        self.daily_data = s.reindex(complete_ts)


def main():
    rbisfiles = glob.glob('/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/01_localstreamflowdata/uuid/data*.txt')
    stationslist = []
    for fpath in rbisfiles:
        test = RBISStation(fpath,{})
        test.read_station_data()
        test.read_daily_data()
        test.get_monthly_data()
        #test.plot_monthly(fpath[:-4] + '.html')
        stationslist.append([fpath[-36:-4], test.station_data['Station identifier'], test.station_data['Station name'],
                             test.station_data['Latitude'], test.station_data['Longitude'],
                             test.station_data['nmonths'], test.station_data['firstmonth'],
                             test.station_data['lastmonth']])
    df = pd.DataFrame(data=stationslist, columns=['dd_id', 'sid', 'stationname', 'lat', 'lon', 'nmonths', 'fmonth', 'lmonth'])
    df.to_csv('/home/home1/gm/projects/DRYvER/03_data/07_streamflowdata/01_localstreamflowdata/uuid/rbisstationinfo.txt',
              index=False)


if __name__ == '__main__':
    main()
