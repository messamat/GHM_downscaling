import os
import requests
import zipfile
import numpy as np
from inspect import getsourcefile
import pandas as pd

from open.helper import get_continental_extent

rootdir = os.path.dirname(os.path.abspath(
    getsourcefile(lambda: 0))).split('\\src')[0]
datadir = os.path.join(rootdir, 'data_wg3')
constants_folder = os.path.join(rootdir, 'src', 'GHM_downscaling', 'constants_wg3')
wginpath = os.path.join(datadir, 'WG3_DATA', 'WG3_STATIC_INPUT')

#Create arcid
continentlist = ['eu']
xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
aoi = ((xmin, xmax), (ymin, ymax))
res = '5min'

write_raster_specs = {
    '30min': 2,
    '5min': 12,
    '6min': 10,
    '30sec': 120,
    '15s': 240
}

rmulti = write_raster_specs[res]
# Get number of cols and rows based on extent and conversion factor
no_cols = round((aoi[0][1] - aoi[0][0]) * rmulti)
no_rows = round((aoi[1][1] - aoi[1][0]) * rmulti)
cellsize = 1 / rmulti
xyarcid_x = np.arange(aoi[0][0]+(cellsize/2.), aoi[0][1]+(cellsize/2.), cellsize).reshape((1,no_cols))
xyarcid_y = np.arange(aoi[1][0]+(cellsize/2.), aoi[1][1]+(cellsize/2.), cellsize).reshape((no_rows, 1))

xyarcid_pd = pd.DataFrame(
    data={'X': np.repeat(xyarcid_x, no_rows, 0).flatten(),
          'Y': np.repeat(xyarcid_y, no_cols, 1).flatten(),
          'arcid': np.arange(no_cols*no_rows, dtype=np.int32)+1,
          'col_ix': np.repeat((np.arange(no_cols)+1).reshape(1,no_cols), no_rows, 0).flatten(),
          'row_ix': np.repeat((np.arange(no_rows)+1).reshape(no_rows,1), no_cols, 1).flatten()}
)

xyarcid_pd.to_csv(os.path.join(constants_folder, 'xyarcid.csv'),
                  index=False)

#Create a text file to write Arcid for land pixels in WG3 grids
class WaterGAPData(object):
    def __init__(self):
        self.landmask = None
        self.data = None
        self.greenland_dropped = False

    def parametrize(self, parameters, constants_folder):
        """
        parametrize the self.data by joining data to it:
        country, continental surface area in the cell, LDD (local drain direction — WaterGAP flow direction)

        :param parameters: additional parameters which should be added to data, either a list or a string
        :return:
        """
        if self.data is None:
            raise Exception('No data read. Please read data first.')

        #Set arcid as the data index
        if isinstance(self.data, pd.Series):
            new_cols = ['arcid', self.data.name]
            self.data = self.data.reset_index()
            self.data.columns = new_cols
        if 'arcid' in self.data.index.names:
            self.data = self.data.reset_index()
        df = self.data

        #Make sure that requested parameter names are correctly formatted
        if isinstance(parameters, str):
            parameters = [parameters]
        implemented_parameters = ['cont_area', 'LDD', 'country']
        for parameter in parameters:
            if parameter not in implemented_parameters:
                raise NotImplementedError

        #Get parameter table and join it to self.data
        if any(['cont_area' in parameters, 'LDD' in parameters, 'country' in parameters]):
            join_df = pd.read_csv(os.path.join(constants_folder,
                                               '{}_ArcID_country_LDD_contarea.txt'.format(self.landmask)),
                                  sep='\t')
            df = df.merge(join_df, left_on='arcid', right_on='Arc_ID', how='right')
            df = df.drop('Arc_ID', axis=1)

        wanted_cols = self.data.columns.tolist() + parameters
        wanted_cols.remove('arcid')
        df = df.set_index('arcid', drop=True)
        self.data = df.loc[:, wanted_cols]
        return self

    def drop_greenland(self, constants_folder):
        if 'country' not in self.data.columns:
            df = self.parametrize(['country'], constants_folder).data
        else:
            df = self.data
        self.data = df[df.country != 304]
        self.greenland_dropped = True
        return self

class Unf(WaterGAPData):  # UNF is the file format of WaterGAP input and output files
    def __init__(self, **kwargs):
        super().__init__()  # Makes Unf inherit methods from "superclass" i.e. WaterGAPData
        self.filename = None
        self.dtype = None
        self.time_step = None
        self.ncols = None
        self.nrows = None
        self.nrows = None
        self.vars = None
        self.unit_time_reference = None
        self.unit_mass = None
        self.unit_area_reference = None
        self.unit = (self.unit_mass, self.unit_area_reference, self.unit_time_reference)
        self.arcid_folderpath = None
        for key in kwargs.keys():
            self.__setattr__(key, kwargs[key])

    def get_dtype(self):
        if self.filename is None:
            raise ValueError('No filename existing.')
        if len(self.filename) < 5:
            raise ValueError('Filename is too short.')
        unf_dtypes = {
            'UNF0': ['>f4', np.float32],
            'UNF1': ['>b', bytes],
            'UNF2': ['>H', int],
            'UNF4': ['>i4', int]
        }
        if self.filename[-4:] in unf_dtypes.keys():
            return unf_dtypes[self.filename[-4:]]
        else:
            raise ValueError('Unf file has not implemented file ending.')

    def get_timestep(self):
        """ Derive timestep and nocolumns by filename

        :return: (timestep, nocolumns)
        """
        time_step_dict = {
            12: 'month',
            365: 'day',
            31: 'day',
            2: 'startend',
            9: 'static'
        }
        splitted_fname = self.filename.split('.')
        if splitted_fname[-2].isdigit():
            n_cols = int(splitted_fname[-2])
            time_step = time_step_dict[n_cols]
        elif splitted_fname[-2][-4:].isdigit():
            time_step = 'year'
            n_cols = 1
        else:
            time_step = 'static'
            n_cols = 1
        return time_step, n_cols

    def analyze_filename(self):
        """
        Analyze filename for further information on dtype, datastructure and additional information

        :return self:
        """
        if self.dtype is None:
            self.dtype = self.get_dtype()
        if self.time_step is None or self.ncols is None:
            self.time_step, self.ncols = self.get_timestep()

        # get additional information
        data_attributes = {}
        split_fname = self.filename.split('.')
        if self.time_step == 'year':
            self.unit_time_reference = 'year'
            data_attributes['year'] = int(split_fname[-2][-4:])
            base_name = split_fname[-2][:-4]
        elif self.time_step == 'month' or self.time_step == 'startend':
            if self.time_step == 'month':
                self.unit_time_reference = 'month'
            data_attributes['year'] = int(split_fname[-3][-4:])
            base_name = split_fname[-3][:-4]
        elif self.time_step == 'day':
            self.unit_time_reference = 'day'
            if self.ncols == 31:
                data_attributes['year'] = int(split_fname[-3].split('_')[-2])
                data_attributes['month'] = int(split_fname[-3].split('_')[-1])
                base_name = '_'.join(split_fname[0].split('_')[:-2])
            else:
                data_attributes['year'] = int(split_fname[-3][-4:])
                base_name = split_fname[-3][:-4]
        elif self.time_step == 'static':
            base_name = split_fname[-2]

        else:
            raise Exception('Something wrent wrong here!')
        return base_name, data_attributes

    def select_arcids(self, arcids):
        """
        Subset grid cells by id (i.e. arcid)

        :param arcids: list of cell ids
        :return self:
        """
        if self.data is None:
            raise Exception('No data read.')
        if arcids == [0]:
            pass
        else:
            self.data = self.data[self.data.arcid.isin(arcids)]
        return self

def simple_read_unf(filepath, arcid_folderpath, **kwargs):
    unf_instance = Unf(arcid_folderpath=arcid_folderpath)
    unf_instance.filename = os.path.split(filepath)[-1]
    if unf_instance.filename is None:
        raise ValueError('No filename existing.')
    if len(unf_instance.filename) < 5:
        raise ValueError('Filename is to short.')
    basename, data_attributes = unf_instance.analyze_filename()

    # read in data
    data = np.fromfile(filepath, dtype=unf_instance.dtype[0]).astype(unf_instance.dtype[1])
    return data

dat_gc= simple_read_unf(os.path.join(wginpath, 'GC.UNF2'), arcid_folderpath=constants_folder)
dat_gr= simple_read_unf(os.path.join(wginpath, 'GR.UNF2'), arcid_folderpath=constants_folder)

ar_wg_mask = np.empty([np.max(dat_gr), np.max(dat_gc)], dtype=int)
for i in range(len(dat_gc)):
    ar_wg_mask[dat_gr[i]-1, dat_gc[i]-1] = i+1

xyarcid_pd_mask = xyarcid_pd.copy()
xyarcid_pd_mask['arcid_wg_mask'] = ar_wg_mask.flatten()
xyarcid_pd_mask_sub = xyarcid_pd_mask[xyarcid_pd_mask['arcid_wg_mask'] !=0][['arcid', 'arcid_wg_mask']]
xyarcid_pd_mask_sub.sort_values(by='arcid_wg_mask')
xyarcid_pd_mask_sub.to_csv(os.path.join(constants_folder, 'slm_arcid.txt'),
                           index=False)


# #Download reference grid from HydroSHEDS
# DDM5_url = "https://data.hydrosheds.org/file/hydrosheds-v1-dir/hyd_glo_dir_5m.zip"
# zip_path_DDM5 = os.path.join(datadir, os.path.split(DDM5_url)[1])
#
# if not os.path.exists(zip_path_DDM5):
#     with open(zip_path_DDM5, "wb") as file:
#         # get request
#         print(f"Downloading DDM5")
#         response = requests.get(DDM5_url, verify=False)
#         file.write(response.content)
# else:
#     print(zip_path_DDM5, "already exists... Skipping download.")
#
# with zipfile.ZipFile(zip_path_DDM5, 'r') as zip_ref:
#     zip_ref.extractall(os.path.dirname(zip_path_DDM5))
#
# #Extract it



