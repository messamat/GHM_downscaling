import pandas as pd
from os import path

from constants.gsim_ewa_lutable import gsimewalutable, gsimafdlutable, gsimgrdblutable
from observation.DailyStation import DailyStationData


class GSIMStation(DailyStationData):
    def __init__(self, station_id, path, station_data):
        super().__init__(station_data)
        self.sid = station_id
        self.gsim_basepath = path
        if station_id in gsimewalutable.keys():
            self.daily_sid = gsimewalutable[station_id]
            self.daily_source = 'ewa'
            self.gsim_basepath = self.gsim_basepath + '02_ewa/'
        elif station_id in gsimafdlutable.keys():
            self.daily_sid = gsimafdlutable[station_id]
            self.daily_source = 'afd'
            self.gsim_basepath = self.gsim_basepath + '03_afd/'
        elif station_id in gsimgrdblutable.keys():
            self.daily_sid = gsimgrdblutable[station_id]
            self.daily_source = 'grdb'
            self.gsim_basepath = self.gsim_basepath + '04_grdb/'
        else:
            print('{} not in daily sources'.format(self.sid))
            self.gsim_basepath = self.gsim_basepath + '06_gsim_monthly/'
            self.daily_sid = 'nan'
            self.daily_source = 'nan'

    def read_daily_data(self):
        try:
            daily_sid = int(self.daily_sid)
        except ValueError:
            return None
        if self.daily_source == 'ewa':
            datapath = '{}{}.day'.format(self.gsim_basepath, daily_sid)
            if path.exists(datapath):
                df = pd.read_csv(datapath,
                                 comment='#',
                                 sep=';',
                                 parse_dates=[1],
                                 engine='python')
                df['date'] = pd.to_datetime(df['YYYY-MM-DD'])
                df = df.set_index(['date'])
                s = df.loc[:, ' Original']
                s = s[s != -999]
                s = s[s != -99]
                complete_ts = pd.date_range(start=s.index.min(), end=s.index.max(), freq='D')
                self.daily_data = s.reindex(complete_ts)
                return self.daily_data
            else:
                print('{} not found'.format(datapath))
                return None
        elif self.daily_source == 'afd':
            datapath = '{}{}.csv'.format(self.gsim_basepath, daily_sid)
            if path.exists(datapath):
                df = pd.read_csv(datapath)
                df['date'] = pd.to_datetime(df['fecha'], format='%d/%m/%Y')
                df = df.set_index(['date'])
                s = df.loc[:, 'caudal']
                complete_ts = pd.date_range(start=s.index.min(), end=s.index.max(), freq='D')
                self.daily_data = s.reindex(complete_ts)
                return self.daily_data
            else:
                print('{} not found'.format(datapath))
                return None
        elif self.daily_source == 'grdb':
            datapath = '{}{}_Q_Day.Cmd.txt'.format(self.gsim_basepath, daily_sid)
            if path.exists(datapath):
                df = pd.read_csv(datapath,
                                 comment='#',
                                 sep=';',
                                 parse_dates=[1],
                                 engine='python')
                df['date'] = pd.to_datetime(df['YYYY-MM-DD'])
                df = df.set_index(['date'])
                df.columns = [x.replace(' ', '') for x in df.columns]
                s = df.loc[:, 'Value']
                s = s[s != -999]
                s = s[s != -99]
                complete_ts = pd.date_range(start=s.index.min(), end=s.index.max(), freq='D')
                self.daily_data = s.reindex(complete_ts)
                return self.daily_data
            pass

    def read_monthly_data(self, calcindex='MEAN'):
        datapath = path.join(self.gsim_basepath, self.sid + '.mon')
        if path.exists(datapath):
            df = pd.read_csv(datapath,
                             comment='#',
                             sep=',\t',
                             parse_dates=[1],
                             engine='python')
            df.columns = [x.replace('"', '') for x in df.columns]
            df.date = pd.to_datetime(df.date)
            df = df.set_index('date')
            # df = df[df["n.missing"] <= df["n.available"]]
            df = df[df["n.missing"] <= 0]
            self.monthly_data = df.loc[:, calcindex]
            self.station_data['nmonths'] = self.monthly_data.shape[0]
            self.station_data['firstmonth'] = self.monthly_data.index[0]
            self.station_data['lastmonth'] = self.monthly_data.index[-1]
            self.station_data['int_calc'] = int(sum(df.loc[:, 'MIN7'] < 0.001) >=1)
            return self
        else:
            return None
