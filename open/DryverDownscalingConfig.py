import pickle
import os

class DownscalingConfig:
    """
    Configuration class for the downscaling

    Parameters
    ----------
    wg_in_path : str
        full path to the WaterGAP input folder
    wg_out_path : str
        full path to the WaterGAP output folder which you want to downscale
    hydrosheds_path : str
        full path to hydrosheds static data
    startyear : int
        startyear of evaluated time period
    endyear : int
        endyear of evaluated time period
    temp_dir : str
        path where the output_raster and results in general are written to
    write_raster : bool, default False
        trigger if temporary raster are written
    mode : {'longterm_avg', 'ts'}, default 'ts'
        mode how Downscaling is run long term average (longterm_avg) and timestep (ts) is implemented
    contintent : {'euassi'}, default 'euassi'
        on which continent or continentcombination you want to process (euassi stands for europe asia and sibira)
    pois : {False, pd.DataFrame}, default False
        if Dataframe provided: points of interest with row and columns for which streamflow is writen into text file
    runoff_src : {'srplusgwr', 'sr', 'cellrunoff', 'totalrunoff'}, default 'srplusgwr'
        the variables the raw runoff calculation should be based on
    correct_global_lakes: bool, default True
        trigger to decide if correction for global lakes and reservoirs should be executed
    sr_smoothing: bool, default False
        trigger to decide if raw runoff should be smoothed and outliers should be removed
    l12harm: bool, default False
        trigger to decide if raw disaggregated should be averaged out over HydroSHEDS level 12 basins
    dis_corr: bool, default True
        trigger to decide if correction with net cell runoff resp. discharge of lr model should be applied
    large_river_corr: bool, default True
        trigger to decide if a correction due to deviating river catchments should be corrected in large rvers
    corr_grid_shift: bool, default True
        trigger to decide if the correction values are partially shifted downstream
    corr_grid_smoothing: bool, default False
        trigger to decide if the correction values are smoothed downstream
    correction_threshold_per_skm: double, default 0.001
        correction_threshold per square kilometer upstream area how much the raw streamflow is allowed to be modified
    aoi: tuple of tuple
        area of interest for which calculations are done
    **kwargs :
        keyword arguments
    """

    def __init__(self,
                 wg_in_path,
                 wg_out_path,
                 hydrosheds_path,
                 startyear,
                 endyear,
                 temp_dir,
                 write_raster=False,
                 write_result='raster',
                 write_dir=None,
                 mode='ts',
                 continent='euassi',
                 pois=False,
                 constants_folder=None,
                 runoff_src='srplusgwr',
                 correct_global_lakes=True,
                 sr_smoothing=False,
                 l12harm=False,
                 dis_corr=True,
                 large_river_corr=True,
                 corr_grid_shift=True,
                 corr_grid_smoothing=False,
                 correction_threshold_per_skm=0.001,
                 **kwargs):

        if mode not in ['longterm_avg', 'ts']:
            Exception('"{}"-mode not implemented'.format(mode))
        self.mode = mode
        self.wg_in_path = wg_in_path
        self.wg_out_path = wg_out_path
        self.hydrosheds_path = hydrosheds_path
        self.startyear = startyear
        self.endyear = endyear
        self.temp_dir = temp_dir
        self.write_raster = write_raster
        self.write_result = write_result
        self.write_dir = write_dir
        self.mode = mode
        self.kwargs = kwargs
        self.continent = continent
        self.pois = pois
        self.constants_folder = constants_folder
        self.runoff_src = runoff_src
        self.correct_global_lakes = correct_global_lakes
        self.sr_smoothing = sr_smoothing
        self.l12harm = l12harm
        self.dis_corr = dis_corr
        self.large_river_corr = large_river_corr
        self.corr_grid_shift = corr_grid_shift
        self.corr_grid_smoothing = corr_grid_smoothing
        self.correction_threshold = correction_threshold_per_skm
        if 'area_of_interest' in kwargs:
            self.aoi = kwargs['area_of_interest']
        else:
            self.aoi = ((-180, 180), (-90, 90))

    def pickle(self, sdir='tempdir'):
        """ Method to pickle the DryverDownscalingConfig to file 'config.pickle'

        Parameters
        ==========
        sdir: str
            directorypath where to write the config.pickle is written to , if 'tempdir' then the self.temp_dir is used
        """
        if sdir == 'tempdir':
            sdir = self.temp_dir
        with open(os.path.join(sdir, 'config.pickle'), 'wb') as f:
            pickle.dump(self, f)
