from os import path
from collections import OrderedDict
from datetime import timedelta
import json


from numpy import nanmean, count_nonzero
import plotly.express as px
import pandas as pd
import numpy as np


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # 👇️ alternatively use str()
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return json.JSONEncoder.default(self, obj)


class DailyStationData:
    def __init__(self, station_data):
        self.daily_data = None
        self.monthly_data = None
        self.dayswithoutflow = None
        self.station_data = station_data

    def read_daily_data(self):
        pass

    def flag_daily_data(self, pathqc):
        if path.exists(f'{pathqc}{self.station_data["dd_id"]}_qc.json'):
            return None
        if self.daily_data is None:
            return None
        qcdict = {'meta': self.station_data,
                  'flagged_values': OrderedDict()}
        flag_df = pd.DataFrame(columns=['ts', 'value', 'reason'])
        # check on negative values
        neg_df = self.daily_data[self.daily_data < 0 ].reset_index()
        neg_df.columns = ['ts', 'value']
        neg_df['reason'] = 'negative value'
        flag_df = pd.concat([flag_df, neg_df])
        # check on daily values with more than 10 consecutive equal discharge values larger than zero
        sameasnextday = ((self.daily_data == self.daily_data.shift()) & (self.daily_data > 0))
        flag_sameday = sameasnextday.groupby(sameasnextday.eq(0).cumsum()).cumcount()
        tendayconsec = self.daily_data[flag_sameday == 9]
        tendayconsec = tendayconsec.reset_index()
        tendayconsec.columns = ['ts', 'value']
        tendayconsec['reason'] = '10day consecutive value'
        flag_df = pd.concat([flag_df, tendayconsec])
        if flag_df.shape[0] == 0:
            return None
        flag_df['maintain'] = 0
        flag_df = flag_df.set_index('ts')
        flag_df.sort_index(inplace=True)
        flag_df.index = [str(x) for x in flag_df.index]
        flag_df.reset_index(inplace=True)
        flag_df = flag_df.rename(columns={'index': 'ts'})
        qcdict['flagged_values'] = flag_df.to_dict('index', into=OrderedDict)
        with open(f'{pathqc}{self.station_data["dd_id"]}_qc.json', "w") as f:
            json.dump(qcdict, f, cls=NpEncoder)

    def clean_daily_data(self, pathqc):
        if not path.exists(f'{pathqc}{self.station_data["dd_id"]}_qc.json'):
            return None
        with open(f'{pathqc}{self.station_data["dd_id"]}_qc.json', "r") as f:
            flag_dc = json.load(f)
        for tsd in flag_dc['flagged_values'].values():
            if tsd['maintain'] == 0:
                ts = pd.Timestamp(tsd['ts'])
                if tsd['reason'] == '10day consecutive value':
                    self.daily_data[pd.date_range(start=ts - timedelta(days=9), end=ts)] = np.nan
                    di = timedelta(days=1)
                    while True:
                        if self.daily_data.index.max() <= ts + di:
                            break
                        if self.daily_data[ts + di] == tsd['value']:
                            self.daily_data[ts + di] = np.nan
                            di += timedelta(days=1)
                        else:
                            break
                elif tsd['reason'] == 'negative value':
                    self.daily_data[ts] = np.nan

    def get_monthly_data(self):
        if self.daily_data is None:
            dd = self.read_daily_data()
            if dd is None:
                return None
        s = self.daily_data.copy()
        below_threshold = (s < 0.001).astype(int)
        sameasnextday = (below_threshold == below_threshold.shift()) & below_threshold
        consecutive_drydays = sameasnextday.groupby(sameasnextday.eq(0).cumsum()).cumcount()
        # mark as intermittent if at least 5 days streamflow below 0.001
        self.station_data['int_calc'] = int(consecutive_drydays.max() >= 4)
        s = s.dropna()
        tempdf = s.resample('M').agg([nanmean, count_nonzero, len])
        tempdf['ndays'] = [x.days_in_month for x in tempdf.index]
        tempdf = tempdf[tempdf['len'] == tempdf['ndays']]
        tempdf['nzeroflow'] = tempdf['ndays'] - tempdf['count_nonzero']
        self.monthly_data = tempdf['nanmean']
        self.monthly_data.name = 'dis'
        self.dayswithoutflow = tempdf['nzeroflow']
        self.dayswithoutflow.name = 'dayswithoutflow'
        self.station_data['nmonths'] = self.monthly_data.shape[0]
        self.station_data['firstmonth'] = self.monthly_data.index[0]
        self.station_data['lastmonth'] = self.monthly_data.index[-1]

        return self.monthly_data

    def plot_monthly(self, outpath):
        if self.monthly_data is None:
            print('monthly data not available')
            return None
        else:
            fig = px.line(y=self.monthly_data, x=self.monthly_data.index,
                          title='{}_{}'.format(self.station_data['Station identifier'],
                                               self.station_data['Station name']))
            fig.update_layout(font_size=24)
            fig.write_html(outpath)
