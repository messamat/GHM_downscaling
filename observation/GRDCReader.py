import pandas as pd
from os import path
from numpy import nan, nanmean

from observation.DailyStation import DailyStationData


class GRDCStation(DailyStationData):

    def __init__(self, station_id, fpath, station_data):
        super().__init__(station_data)
        self.sid = station_id
        self.grdc_basepath = fpath

    def read_daily_data(self):
        datapath = '{}{}.txt'.format(self.grdc_basepath + '01_grdc/', self.sid)
        if path.exists(datapath):
            df = pd.read_csv(datapath,
                             comment='#',
                             sep=';',
                             parse_dates=[1],
                             engine='python')
            df['date'] = pd.to_datetime(df['YYYY-MM-DD'])
            df = df.set_index(['date'])
            s = df.loc[:, ' Calculated']
            s = s[s != -999]
            s = s[s != -99]
            complete_ts = pd.date_range(start=s.index.min(), end=s.index.max(), freq='D')
            self.daily_data = s.reindex(complete_ts)
            return self.daily_data
        else:
            return None


def main():
    test = GRDCStation('6990500', '/home/home8/dryver/observed_messager/forDRYVER20210316/GRDCdat_day/')
    a = test.get_monthly_data()
    print('debug')


if __name__ == '__main__':
    main()
