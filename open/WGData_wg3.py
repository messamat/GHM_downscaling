from datetime import datetime
from open.DownScaleArray import DownScaleArray

#RunDownscaling.py #####################################################################################################
import os
from inspect import getsourcefile
from osgeo import gdal, osr, ogr
import pandas as pd
from codetiming import Timer

from open.DryverDownscalingWrapper import DryverDownscalingWrapper, gather_finished_downscaling, run_prepared_downscaling
from open.DryverDownscalingConfig import DownscalingConfig
from open.helper import get_continental_extent

rootdir = os.path.dirname(os.path.abspath(
    getsourcefile(lambda: 0))).split('\\src')[0]
datadir = os.path.join(rootdir, 'data_wg3')
localdir = os.path.join(rootdir, 'results_wg3', 'downscaling_output_{}'.format(datetime.today().strftime('%Y%m%d')))
if not os.path.exists(localdir):
    os.mkdir(localdir)

continentlist = ['eu']  # ['eu', 'as', 'si']
continent = ''.join(continentlist)
wginpath = os.path.join(datadir, 'WG3_DATA', 'WG3_STATIC_INPUT')  # '/home/home1/gm/datasets/input_routing/wghm22e_v001/input/'
wgpath = os.path.join(datadir, 'WG3_DATA', 'WG3_OUTPUT')  # '/home/home8/dryver/22eant/'
hydrosheds_folder = os.path.join(datadir,
                                 'hs_reproduced')  # '/home/home1/gm/projects/DRYvER/03_data/12_downscalingdata_eu/'
setup_folder = os.path.join(datadir, 'setupdata_for_downscaling')
stations_path = os.path.join(setup_folder, 'stations.csv')
constants_folder = os.path.join(rootdir, 'src', 'GHM_downscaling', 'constants_wg3')
pois = pd.read_csv(stations_path)  # points of interest
if continent in {'eu', 'as', 'si', 'sa'}:
    xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
    aoi = ((xmin, xmax), (ymin, ymax))
if continent == 'rhone':
    aoi = ((3.5, 9), (43, 48.5))


dconfig = DownscalingConfig(wg_in_path=wginpath,
                            wg_out_path=wgpath,
                            hydrosheds_path=hydrosheds_folder,
                            startyear=2003,
                            endyear=2004,
                            temp_dir=localdir,
                            write_raster=True,
                            write_result='nc',
                            write_dir=localdir,
                            mode='ts',
                            continent=continent,
                            constants_folder=constants_folder,
                            pois=pois,
                            runoff_src='srplusgwr',
                            correct_global_lakes=True,
                            sr_smoothing=False,
                            l12harm=True,
                            dis_corr=True,
                            large_river_corr=True,
                            corr_grid_shift=True,
                            corr_grid_smoothing=False,
                            correction_threshold_per_skm=0.001,
                            area_of_interest=aoi,
                            # corrweightfactor=0.1
                            )
#DryverDownscalingWrapper ##############################################################################################
import pickle
import glob
from multiprocessing import Pool
from functools import partial
import os

import numpy as np
import pandas as pd
import osgeo.gdal
from codetiming import Timer

from open.WGData import WGData
from open.HydroSHEDSData import HydroSHEDSData
from open.DownScaleArray import DownScaleArray
from open.DryverDownscaling import run_task

config=dconfig #################### ADDED
kwargs = dict()

#--------------------------- Run class definition: down = DryverDownscalingWrapper(dconfig) ---------------------------------------------------------------------
with open(os.path.join(config.temp_dir, 'run_information.txt'), 'w') as f:
    temp = vars(config)
    for item in temp:
        f.write('{} : {}\n'.format(item, temp[item]))

mode = config.mode
dconfig = config
kwargs.update(config.kwargs)
kwargs = kwargs

#------------------------- RUN WGData --------------------------------------------------------------------------------
#wg = WGData(config=dconfig, **kwargs) #Get WaterGAP object instance (data and tools related to WG)
coords = pd.read_csv(os.path.join(config.constants_folder, 'xyarcid.csv'))

read_variable_kwargs = {
    "wg_simoutput_path": config.wg_out_path,
    "timestep": 'month',
    "startyear": config.startyear,
    "endyear": config.endyear
}

if "area_of_interest" in kwargs:
    aoi = kwargs['area_of_interest']
    arcids = coords.set_index(["X", "Y"])
    relative_arcids_df = arcids.sort_index().loc[(slice(aoi[0][0], aoi[0][1]),
                                                  slice(aoi[1][0], aoi[1][1])), 'arcid']
    relative_arcids = relative_arcids_df.values.tolist()
    read_variable_kwargs['arcid_list'] = relative_arcids
    coords = relative_arcids_df.reset_index().set_index('arcid')
else:
    aoi = ((-180., 180.), (-90., 90.))
    coords = coords.set_index('arcid')

# Read data and prepare paths -----------------------------------------------------------------------------------
# wg_input = InputDir().read(in_path=config.wg_in_path,
#                                 arcid_folderpath=config.constants_folder,
#                                 explain=False,
#                                 files=['G_FLOWDIR.UNF2', 'GR.UNF2', 'GC.UNF2', 'GCONTFREQ.UNF0'],
#                                 additional_files=False, cellarea=True)
in_path=config.wg_in_path
arcid_folderpath=config.constants_folder
explain=False
files=['G_FLOWDIR.UNF2', 'GR.UNF2', 'GC.UNF2', 'GCONTFREQ.UNF0']
additional_files=False
cellarea=True
drop_greenland=False

if files == 'all':
    relevant_files = [
        # "GALTMOD.UNF0",
        "G_AQ_FACTOR.UNF1",
        "G_ARID_HUMID.UNF2",
        "G_BANKFULL.UNF0",
        "GBUILTUP.UNF0",
        "GCONTFREQ.UNF0",  ######### MUST HAVE BUT MISSING
        "GCOUNTRY.UNF2",
        # "GCRC.UNF4",
        "GC.UNF2",  ######### MUST HAVE BUT MISSING
        # "G_ELEV_RANGE.101.UNF2",
        "G_FLOWDIR.UNF2",
        # "G_FRACTRETURNGW_IRRIG.UNF0",
        "G_GLOLAK.UNF0",
        "G_GLOWET.UNF0",
        # "G_GW_FACTOR_CORR.UNF0",
        "G_LAKAREA.UNF0",
        # "G_LANDCOVER.UNF1",
        "G_LOCLAK.UNF0",
        "G_LOCRES.UNF0",
        # "G_LOCWET.UNF0",
        "GLWDunits.UNF4",
        "G_MEANDERING_RATIO.UNF0",
        # "G_MEAN_OUTFLOW.12.UNF0",
        # "G_MEAN_OUTFLOW.UNF0",
        "G_NUs_1971_2000.UNF0",  ######TO RENAME
        # "G_OUTFLOW_CELL_ASSIGNMENT.UNF4",
        "G_PERMAGLAC.UNF1",
        # "G_REGLAKE.UNF0",
        # "G_REG_LAKE.UNF1",
        "G_RESAREA.UNF0",  #####unf4
        "G_RES_TYPE.UNF1",
        "G_ROUGHNESS.UNF0",
        "GR.UNF2",
        "G_SLOPE_CLASS.UNF1",
        # "G_START_YEAR.UNF4",
        "G_STORAGE_CAPACITY.UNF0",
        # "G_TAWC.UNF0",
        "G_TEXTURE.UNF1"]
else:
    relevant_files = files

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

    def append_unf(self, other, verify_integrity=False, **kwargs):
        """
        appends a unf file of the same variable to each other

        :param other: path of unf file which should be appended
        :param verify_integrity: determine if you want check with indices that you have no doubled values
        :return: self
        """
        if isinstance(other, Unf):
            other_unf = other
        else:
            other_unf = read_unf_file(filepath=other,
                                      arcid_folderpath=self.arcid_folderpath,
                                      **kwargs)

        if (self.time_step != other_unf.time_step
                or self.nrows != other_unf.nrows
                or self.ncols != other_unf.ncols
                or self.vars != other_unf.vars):
            raise Exception('Other Unffile doesnt fit to the first one')

        if verify_integrity:
            a = self.data.set_index([x for x in self.data.columns if x != self.vars[0]])
            b = other_unf.data.set_index([x for x in other_unf.data.columns if x != other_unf.vars[0]])
            self.data = pd.concat([a, b], verify_integrity=True).reset_index()
        else:
            self.data = pd.concat([self.data, other_unf.data])

        return self

#######################################################
# x = read_unf_file(os.path.join(in_path, inputfile),
#                   arcid_folderpath=arcid_folderpath)
inputfile='G_FLOWDIR.UNF2'
filepath=os.path.join(in_path, inputfile)

days_in_month = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}

if 'landmask_ref' not in kwargs.keys(): ########################################################################DOUBLE CHECK HOW TO DEAL WITH IT
    #By default, set landmask_ref to WG2 characteristics
    landmask_ref = {
        180721: 'slm' #number of cells in DDM5 routing network landmask
        # 66896: 'slm', #number of cells in DDM30 routing network landmask
        # 67420: 'wlm' #number of cells in CRU landmask (the WG climate forcing data landmask)
    }

var_dict = {
    'G_RIVER_AVAIL_': {'var_name': 'dis', #river discharge in grid cell, km³/month
                       'mass_unit': None,
                       'area_unit': None},
    'G_RIVER_AVAIL': {'var_name': 'dis',
                      'mass_unit': None,
                      'area_unit': None},
    'G_RIVER_IN_UPSTREAM_': {'var_name': 'dis_upstream', #river discharge in upstream grid cell, km³/month
                             'mass_unit': None,
                             'area_unit': None},
    'G_PRECIPITATION_': {'var_name': 'pr',
                         'mass_unit': 'kg',
                         'area_unit': 'm2'},
    'G_CONSISTENT_PRECIPITATION_km3_': {'var_name': 'precip',
                                        'mass_unit': None,
                                        'area_unit': None},
    'G_FLOWDIR': {'var_name': 'flowdir',
                  'mass_unit': None,
                  'area_unit': None},
    'G_BASINS': {'var_name': 'basins',
                 'mass_unit': None,
                 'area_unit': None},
    'G_BASINS_2': {'var_name': 'basins2',
                   'mass_unit': None,
                   'area_unit': None},
    'GPREC': {'var_name': 'precip',
              'mass_unit': None,
              'area_unit': None},
    'G_TEMPERATURE_': {'var_name': 'temperature',
                       'mass_unit': None,
                       'area_unit': None},
    'GTEMP': {'var_name': 'temperature',
              'mass_unit': None,
              'area_unit': None},
    'G_SOIL_WATER_': {'var_name': 'rootmoist',
                      'mass_unit': None,
                      'area_unit': None},
    'G_SNOW_WATER_EQUIV_': {'var_name': 'swe',
                            'mass_unit': None,
                            'area_unit': None},
    'G_SNOW_FALL_': {'var_name': 'psnow',
                     'mass_unit': None,
                     'area_unit': None},
    'G_CELL_RUNOFF_mm_': {'var_name': 'net_cell_runoff', #total cell runoff (outflow of lakes/wetlands); actual inflow into stream
                          'mass_unit': 'mm',
                          'area_unit': 'm2'},
    'G_CELL_RUNOFF_': {'var_name': 'net_cell_runoff',
                       'mass_unit': None,
                       'area_unit': None},
    'G_NETUSE_GW_m3_': {'var_name': 'na_g',
                        'mass_unit': None,
                        'area_unit': None},
    'G_NETUSE_SW_m3_': {'var_name': 'na_s',
                        'mass_unit': None,
                        'area_unit': None},
    'G_ACTUAL_WATER_CONSUMPTION_': {'var_name': 'actual_water_use',
                                    'mass_unit': None,
                                    'area_unit': None},
    'G_INFLC.9': {'var_name': 'infl_arcid',
                  'mass_unit': None,
                  'area_unit': None},
    'G_ACTUAL_NAS_': {'var_name': 'actual_use_sw',
                      'mass_unit': None,
                      'area_unit': None},
    'G_ACTUAL_NAG_': {'var_name': 'actual_use_gw',
                      'mass_unit': None,
                      'area_unit': None},
    'G_CELL_AET_': {'var_name': 'cell_aet',
                    'mass_unit': 'mm',
                    'area_unit': 'm2'},
    'G_CELLAET_CONSUSE_km3_': {'var_name': 'cell_aet_consuse',
                               'mass_unit': 'km3',
                               'area_unit': 'contarea'},
    'G_TOTAL_PET_': {'var_name': 'total_pet',
                     'mass_unit': None,
                     'area_unit': None},
    'G_TOTAL_STORAGES_STARTEND_km3_': {'var_name': 'tws_startend',
                                       'mass_unit': None,
                                       'area_unit': None},
    'G_SATIS_USE_': {'var_name': 'statisfied_use',
                     'mass_unit': None,
                     'area_unit': None},
    'G_IRRIG_WITHDRAWAL_USE_HISTAREA_m3_': {'var_name': 'irrig_withdrawal_use',
                                            'mass_unit': None,
                                            'area_unit': None},
    'G_IRRIG_CONS_USE_HISTAREA_m3_': {'var_name': 'irrig_cons_use',
                                      'mass_unit': None,
                                      'area_unit': None},
    'G_FRACTRETURNGW_IRRIG': {'var_name': 'frig',
                              'mass_unit': None,
                              'area_unit': None},
    'G_UNSAT_USE_': {'var_name': 'unsat_use',
                     'mass_unit': None,
                     'area_unit': None},
    'G_GLO_WETL_EXTENT_km2_': {'var_name': 'glo_wetl_extent',
                               'mass_unit': None,
                               'area_unit': None},
    'G_TOTAL_GW_RECHARGE_km3_': {'var_name': 'total_gw_recharge',
                                 'mass_unit': None,
                                 'area_unit': None},
    'G_ALLOC_USE_': {'var_name': 'allocated_use',
                     'mass_unit': None,
                     'area_unit': None},
    'G_SATIS_ALLOC_USE_IN_2NDCELL_': {'var_name': 'alloc_use_in2ndcell',
                                      'mass_unit': None,
                                      'area_unit': None},
    'G_CONT_AREA_OUT_km2': {'var_name': 'contarea',
                            'mass_unit': None,
                            'area_unit': None},
    'G_TOTAL_GW_RECHARGE_mm_': {'var_name': 'total_gw_recharge',
                                'mass_unit': None,
                                'area_unit': None},
    'G_GW_RECHARGE_km3_': {'var_name': 'qr',
                           'mass_unit': 'km3',
                           'area_unit': None},
    'G_GW_RECHARGE_mm_': {'var_name': 'qr',
                          'mass_unit': 'mm',
                          'area_unit': 'm2'},
    'G_GWR_SURFACE_WATER_BODIES_km3_': {'var_name': 'qr',
                                        'mass_unit': 'km3',
                                        'area_unit': 'contarea'},
    'G_TOTAL_STORAGES_mm_': {'var_name': 'tws',
                             'mass_unit': 'mm',
                             'area_unit': 'm2'},
    'G_TOTAL_CONS_USE_GW_HISTAREA_m3_': {'var_name': 'cons_use_gw',
                                         'mass_unit': None,
                                         'area_unit': None},
    'G_TOTAL_WITHDRAWAL_USE_GW_HISTAREA_m3_': {'var_name': 'withdrawal_gw',
                                               'mass_unit': None,
                                               'area_unit': None},
    'G_LAND_AREA_FRACTIONS_': {'var_name': 'landareafr',
                               'mass_unit': None,
                               'area_unit': None},
    'G_GROUND_WATER_STORAGE_mm_': {'var_name': 'gws',
                                   'mass_unit': 'mm',
                                   'area_unit': 'm2'},
    'G_SOIL_WATER_STORAGE_mm_': {'var_name': 'sws',
                                 'mass_unit': 'mm',
                                 'area_unit': 'm2'},
    'G_GLOWET': {'var_name': 'glwd_glowet',
                 'mass_unit': None,
                 'area_unit': None},
    'G_LOCWET': {'var_name': 'glwd_locwet',
                 'mass_unit': None,
                 'area_unit': None}
}

unf_instance = Unf(arcid_folderpath=arcid_folderpath)
unf_instance.filename = os.path.split(filepath)[-1]
if unf_instance.filename is None:
    raise ValueError('No filename existing.')
if len(unf_instance.filename) < 5:
    raise ValueError('Filename is to short.')
basename, data_attributes = unf_instance.analyze_filename()

# read in data
data = np.fromfile(filepath, dtype=unf_instance.dtype[0]).astype(unf_instance.dtype[1])
unf_instance.nrows = int(data.shape[0] / unf_instance.ncols) #wg data are unidimensional
unf_instance.landmask = landmask_ref[unf_instance.nrows]

# get arcids and cast to two-dimensional data.frame ########################################################################DOUBLE CHECK HOW TO DEAL WITH IT
arcid_ref = pd.read_csv(os.path.join(arcid_folderpath,
                                     landmask_ref[unf_instance.nrows] +
                                     '_arcid.txt'))
data = pd.DataFrame(data.reshape((unf_instance.nrows, unf_instance.ncols)),
                    columns=range(1, 1 + unf_instance.ncols))

# restrict to days in month
if unf_instance.ncols == 31:
    data = data.iloc[:, :days_in_month[data_attributes['month']]]

#Add ArcID column
data = data.assign(arcid=arcid_ref.loc[:, 'arcid'])

#If dimension does not match corresponding landmask (DDM30 or CRU), remove NAs
if unf_instance.nrows not in landmask_ref.keys():
    data.dropna(inplace=True)

#Identify variable name
try:
    var_name = var_dict[basename]['var_name']
except KeyError:
    var_name = 'variable'

#If time series data, format to long form
if unf_instance.time_step == 'static':
    data.columns = [var_name, 'arcid']
    unf_instance.data = data
elif unf_instance.time_step is not None:
    data = pd.melt(data, id_vars=['arcid'],
                   var_name=unf_instance.time_step,
                   value_name=var_name)
    if unf_instance.ncols == 31:
        data = data.assign(month=int(data_attributes['month']))
    data = data.assign(year=int(data_attributes['year']))

    # dtype conversion
    dtype_dict = {col[0]: col[1] for col in [('arcid', 'int32'),
                                             ('month', 'int8'),
                                             ('year', 'int16'),
                                             ('day', 'int8')] if col[0] in data.columns}
    data = data.astype(dtype_dict)

    #Assign resulting data to unf_instance
    unf_instance.data = data
else:
    raise ValueError('No timestep given')
if var_name != 'variable':
    unf_instance.vars = [var_dict[basename]['var_name']]
    unf_instance.unit_area_reference = var_dict[basename]['area_unit']
    unf_instance.unit_mass = var_dict[basename]['mass_unit']
    unf_instance.unit = (var_dict[basename]['mass_unit'],
                         var_dict[basename]['area_unit'],
                         unf_instance.unit_time_reference)

if 'select_arcids' in kwargs.keys():
    unf_instance.select_arcids(kwargs['select_arcids'])



#######
inputdata = pd.DataFrame()
#for inputfile in relevant_files:
inputfile='GCOUNTRY.UNF2'
x = read_unf_file(os.path.join(in_path, inputfile),
                  arcid_folderpath=arcid_folderpath)
if drop_greenland:
    x = x.drop_greenland(constants_folder=arcid_folderpath)
    x = x.data.iloc[:, 0]
else:
    x = x.data.set_index('arcid').iloc[:, 0]

if inputfile in explanation.keys() and explain:
    x.name = explanation[inputfile][0]
else:
    x.name = inputfile
inputdata = inputdata.merge(x, left_index=True, right_index=True, how='outer')

if additional_files:
    cellarea = True
    elev_range = True
else:
    if 'cellarea' in kwargs:
        cellarea = kwargs['cellarea']
    else:
        cellarea = False

    if 'elev_range' in kwargs:
        elev_range = kwargs['elev_range']
    else:
        elev_range = False

if cellarea:
    garea = np.fromfile(os.path.join(in_path, 'GAREA.UNF0'), dtype='>f4').astype(float)
    inputdata['cellarea'] = [garea[x - 1] for x in inputdata['GR.UNF2']]

if elev_range:
    elev_range = np.fromfile(os.path.join(in_path, "G_ELEV_RANGE.101.UNF2"), dtype='>H').astype(int)

    elev_range = pd.DataFrame(elev_range.reshape((inputdata.shape[0], 101)),
                              columns=['elev_range_{}'.format(x) for x in range(1, 102)])
    inputdata['elev_range'] = elev_range.values.tolist()
# self.data = pd.concat([inputdata.reset_index(), elev_range], axis=1)

if 'select_arcids' in kwargs:
    inputdata = inputdata.loc[kwargs['select_arcids'], :]
self.data = inputdata
return self



#######################################################################################



inputdata = pd.DataFrame()
# for inputfile in relevant_files:
#     x = read_unf_file(os.path.join(in_path, inputfile),
#                       arcid_folderpath=arcid_folderpath)

filepath = os.path.join(in_path, inputfile)

days_in_month = {
    1: 31,
    2: 28,
    3: 31,
    4: 30,
    5: 31,
    6: 30,
    7: 31,
    8: 31,
    9: 30,
    10: 31,
    11: 30,
    12: 31
}






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
    #unf_instance.nrows = int(data.shape[0] / unf_instance.ncols)  # wg data are unidimensional



def read_unf_file(filepath, arcid_folderpath, **kwargs):
    """
    Reading in unf files into an unf object

    :param filepath:
    :param kwargs:
    :return:
    """

    days_in_month = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31
    }

    if 'landmask_ref' not in kwargs.keys():
        #By default, set landmask_ref to WG2 characteristics
        landmask_ref = {
            66896: 'slm', #number of cells in DDM30 routing network landmask
            67420: 'wlm' #number of cells in CRU landmask (the WG climate forcing data landmask)
        }

    var_dict = {
        'G_RIVER_AVAIL_': {'var_name': 'dis', #river discharge in grid cell, km³/month
                           'mass_unit': None,
                           'area_unit': None},
        'G_RIVER_AVAIL': {'var_name': 'dis',
                          'mass_unit': None,
                          'area_unit': None},
        'G_RIVER_IN_UPSTREAM_': {'var_name': 'dis_upstream', #river discharge in upstream grid cell, km³/month
                                 'mass_unit': None,
                                 'area_unit': None},
        'G_PRECIPITATION_': {'var_name': 'pr',
                             'mass_unit': 'kg',
                             'area_unit': 'm2'},
        'G_CONSISTENT_PRECIPITATION_km3_': {'var_name': 'precip',
                                            'mass_unit': None,
                                            'area_unit': None},
        'G_FLOWDIR': {'var_name': 'flowdir',
                      'mass_unit': None,
                      'area_unit': None},
        'G_BASINS': {'var_name': 'basins',
                     'mass_unit': None,
                     'area_unit': None},
        'G_BASINS_2': {'var_name': 'basins2',
                       'mass_unit': None,
                       'area_unit': None},
        'GPREC': {'var_name': 'precip',
                  'mass_unit': None,
                  'area_unit': None},
        'G_TEMPERATURE_': {'var_name': 'temperature',
                           'mass_unit': None,
                           'area_unit': None},
        'GTEMP': {'var_name': 'temperature',
                  'mass_unit': None,
                  'area_unit': None},
        'G_SOIL_WATER_': {'var_name': 'rootmoist',
                          'mass_unit': None,
                          'area_unit': None},
        'G_SNOW_WATER_EQUIV_': {'var_name': 'swe',
                                'mass_unit': None,
                                'area_unit': None},
        'G_SNOW_FALL_': {'var_name': 'psnow',
                         'mass_unit': None,
                         'area_unit': None},
        'G_CELL_RUNOFF_mm_': {'var_name': 'net_cell_runoff', #total cell runoff (outflow of lakes/wetlands); actual inflow into stream
                              'mass_unit': 'mm',
                              'area_unit': 'm2'},
        'G_CELL_RUNOFF_': {'var_name': 'net_cell_runoff',
                           'mass_unit': None,
                           'area_unit': None},
        'G_NETUSE_GW_m3_': {'var_name': 'na_g',
                            'mass_unit': None,
                            'area_unit': None},
        'G_NETUSE_SW_m3_': {'var_name': 'na_s',
                            'mass_unit': None,
                            'area_unit': None},
        'G_ACTUAL_WATER_CONSUMPTION_': {'var_name': 'actual_water_use',
                                        'mass_unit': None,
                                        'area_unit': None},
        'G_INFLC.9': {'var_name': 'infl_arcid',
                      'mass_unit': None,
                      'area_unit': None},
        'G_ACTUAL_NAS_': {'var_name': 'actual_use_sw',
                          'mass_unit': None,
                          'area_unit': None},
        'G_ACTUAL_NAG_': {'var_name': 'actual_use_gw',
                          'mass_unit': None,
                          'area_unit': None},
        'G_CELL_AET_': {'var_name': 'cell_aet',
                        'mass_unit': 'mm',
                        'area_unit': 'm2'},
        'G_CELLAET_CONSUSE_km3_': {'var_name': 'cell_aet_consuse',
                                   'mass_unit': 'km3',
                                   'area_unit': 'contarea'},
        'G_TOTAL_PET_': {'var_name': 'total_pet',
                         'mass_unit': None,
                         'area_unit': None},
        'G_TOTAL_STORAGES_STARTEND_km3_': {'var_name': 'tws_startend',
                                           'mass_unit': None,
                                           'area_unit': None},
        'G_SATIS_USE_': {'var_name': 'statisfied_use',
                         'mass_unit': None,
                         'area_unit': None},
        'G_IRRIG_WITHDRAWAL_USE_HISTAREA_m3_': {'var_name': 'irrig_withdrawal_use',
                                                'mass_unit': None,
                                                'area_unit': None},
        'G_IRRIG_CONS_USE_HISTAREA_m3_': {'var_name': 'irrig_cons_use',
                                          'mass_unit': None,
                                          'area_unit': None},
        'G_FRACTRETURNGW_IRRIG': {'var_name': 'frig',
                                  'mass_unit': None,
                                  'area_unit': None},
        'G_UNSAT_USE_': {'var_name': 'unsat_use',
                         'mass_unit': None,
                         'area_unit': None},
        'G_GLO_WETL_EXTENT_km2_': {'var_name': 'glo_wetl_extent',
                                   'mass_unit': None,
                                   'area_unit': None},
        'G_TOTAL_GW_RECHARGE_km3_': {'var_name': 'total_gw_recharge',
                                     'mass_unit': None,
                                     'area_unit': None},
        'G_ALLOC_USE_': {'var_name': 'allocated_use',
                         'mass_unit': None,
                         'area_unit': None},
        'G_SATIS_ALLOC_USE_IN_2NDCELL_': {'var_name': 'alloc_use_in2ndcell',
                                          'mass_unit': None,
                                          'area_unit': None},
        'G_CONT_AREA_OUT_km2': {'var_name': 'contarea',
                                'mass_unit': None,
                                'area_unit': None},
        'G_TOTAL_GW_RECHARGE_mm_': {'var_name': 'total_gw_recharge',
                                    'mass_unit': None,
                                    'area_unit': None},
        'G_GW_RECHARGE_km3_': {'var_name': 'qr',
                               'mass_unit': 'km3',
                               'area_unit': None},
        'G_GW_RECHARGE_mm_': {'var_name': 'qr',
                              'mass_unit': 'mm',
                              'area_unit': 'm2'},
        'G_GWR_SURFACE_WATER_BODIES_km3_': {'var_name': 'qr',
                                            'mass_unit': 'km3',
                                            'area_unit': 'contarea'},
        'G_TOTAL_STORAGES_mm_': {'var_name': 'tws',
                                 'mass_unit': 'mm',
                                 'area_unit': 'm2'},
        'G_TOTAL_CONS_USE_GW_HISTAREA_m3_': {'var_name': 'cons_use_gw',
                                             'mass_unit': None,
                                             'area_unit': None},
        'G_TOTAL_WITHDRAWAL_USE_GW_HISTAREA_m3_': {'var_name': 'withdrawal_gw',
                                                   'mass_unit': None,
                                                   'area_unit': None},
        'G_LAND_AREA_FRACTIONS_': {'var_name': 'landareafr',
                                   'mass_unit': None,
                                   'area_unit': None},
        'G_GROUND_WATER_STORAGE_mm_': {'var_name': 'gws',
                                       'mass_unit': 'mm',
                                       'area_unit': 'm2'},
        'G_SOIL_WATER_STORAGE_mm_': {'var_name': 'sws',
                                     'mass_unit': 'mm',
                                     'area_unit': 'm2'},
        'G_GLOWET': {'var_name': 'glwd_glowet',
                     'mass_unit': None,
                     'area_unit': None},
        'G_LOCWET': {'var_name': 'glwd_locwet',
                     'mass_unit': None,
                     'area_unit': None}
    }

    unf_instance = Unf(arcid_folderpath=arcid_folderpath)
    unf_instance.filename = os.path.split(filepath)[-1]
    if unf_instance.filename is None:
        raise ValueError('No filename existing.')
    if len(unf_instance.filename) < 5:
        raise ValueError('Filename is to short.')
    basename, data_attributes = unf_instance.analyze_filename()

    # read in data
    data = np.fromfile(filepath, dtype=unf_instance.dtype[0]).astype(unf_instance.dtype[1])
    unf_instance.nrows = int(data.shape[0] / unf_instance.ncols) #wg data are unidimensional
    unf_instance.landmask = landmask_ref[unf_instance.nrows]

    # get arcids and cast to two-dimensional data.frame
    arcid_ref = pd.read_csv(os.path.join(arcid_folderpath,
                                         landmask_ref[unf_instance.nrows] +
                                         '_arcid_gcrc.txt'),
                            sep='\t')
    data = pd.DataFrame(data.reshape((unf_instance.nrows, unf_instance.ncols)),
                        columns=range(1, 1 + unf_instance.ncols))

    # restrict to days in month
    if unf_instance.ncols == 31:
        data = data.iloc[:, :days_in_month[data_attributes['month']]]

    #Add ArcID column
    data = data.assign(arcid=arcid_ref.loc[:, 'ArcID'])

    #If dimension does not match corresponding landmask (DDM30 or CRU), remove NAs
    if unf_instance.nrows not in landmask_ref.values():
        data.dropna(inplace=True)

    #Identify variable name
    try:
        var_name = var_dict[basename]['var_name']
    except KeyError:
        var_name = 'variable'

    #If time series data, format to long form
    if unf_instance.time_step == 'static':
        data.columns = [var_name, 'arcid']
        unf_instance.data = data
    elif unf_instance.time_step is not None:
        data = pd.melt(data, id_vars=['arcid'],
                       var_name=unf_instance.time_step,
                       value_name=var_name)
        if unf_instance.ncols == 31:
            data = data.assign(month=int(data_attributes['month']))
        data = data.assign(year=int(data_attributes['year']))

        # dtype conversion
        dtype_dict = {col[0]: col[1] for col in [('arcid', 'int32'),
                                                 ('month', 'int8'),
                                                 ('year', 'int16'),
                                                 ('day', 'int8')] if col[0] in data.columns}
        data = data.astype(dtype_dict)

        #Assign resulting data to unf_instance
        unf_instance.data = data
    else:
        raise ValueError('No timestep given')
    if var_name != 'variable':
        unf_instance.vars = [var_dict[basename]['var_name']]
        unf_instance.unit_area_reference = var_dict[basename]['area_unit']
        unf_instance.unit_mass = var_dict[basename]['mass_unit']
        unf_instance.unit = (var_dict[basename]['mass_unit'],
                             var_dict[basename]['area_unit'],
                             unf_instance.unit_time_reference)

    if 'select_arcids' in kwargs.keys():
        unf_instance.select_arcids(kwargs['select_arcids'])

    return unf_instance




continentalarea_to_landarea_conversion_factor = None
surface_runoff_land_mm = None
self.land_fractions = read_variable(var='G_LAND_AREA_FRACTIONS_', arcid_folderpath=config.constants_folder,
                                    **read_variable_kwargs)
self.surface_runoff = read_variable(var='G_SURFACE_RUNOFF_', arcid_folderpath=config.constants_folder,
                                    **read_variable_kwargs)
self.total_runoff = read_variable(var='G_RUNOFF_', arcid_folderpath=config.constants_folder,
                                  **read_variable_kwargs)
self.gw_runoff = read_variable(var='G_GW_RUNOFF_mm_', arcid_folderpath=config.constants_folder,
                               **read_variable_kwargs)
self.dis = read_variable(var='G_RIVER_AVAIL_', arcid_folderpath=config.constants_folder,
                         **read_variable_kwargs)  # km3 / month


