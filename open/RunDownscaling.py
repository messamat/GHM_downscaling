import os
from inspect import getsourcefile
import pandas as pd
from codetiming import Timer

from open.DryverDownscalingWrapper import DryverDownscalingWrapper, gather_finished_downscaling, run_prepared_downscaling
from open.DryverDownscalingConfig import DownscalingConfig
from open.helper import get_continental_extent

@Timer(name='decorator', text='Downscaling takes currently {minutes:.0f} minutes')
def main(rtype, rootdir):
    """
    Preparation and running script for Downscaling.
    Modify variables in this main function to configure the downscaling and corresponding paths.
    For description of configuration parameters, please consult DryverDownscalingConfig class.

    Parameters
    ----------
    rtype: str one of ['ipg80', 'ipg50', 'clusterfull']
        System for which the preparation of the downscaling should be conducted and
         in case of ipg80 and ipg50 on which the downscaling should happen

    Returns
    -------

    """
    continentlist = ['eu', 'as', 'si']
    continent = ''.join(continentlist)
    wginpath = os.path.join(rootdir, 'data', 'WG_inout_downscaling_data',
                            'wghm22e_v001', 'input') #'/home/home1/gm/datasets/input_routing/wghm22e_v001/input/'
    wgpath = os.path.join(rootdir, 'data', 'WG_inout_downscaling_data', '22eant') #'/home/home8/dryver/22eant/'
    hydrosheds_path = os.path.join(rootdir, 'data', 'setupdata_for_downscaling') #'/home/home1/gm/projects/DRYvER/03_data/12_downscalingdata_eu/'
    stations_path = os.path.join(hydrosheds_path, 'stations.csv')
    pois = pd.read_csv(stations_path) #points of interest

    xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
    aoi = ((xmin, xmax), (ymin, ymax))
    if continent == 'rhone':
        aoi = ((3.5, 9), (43, 48.5))

    dconfig = DownscalingConfig(wg_in_path=wginpath,
                                wg_out_path=wgpath,
                                hydrosheds_path=hydrosheds_path,
                                startyear=1901,
                                endyear=2019,
                                temp_dir=localdir,
                                write_raster=False,
                                write_result='nc',
                                mode='ts',
                                continent=continent,
                                pois=pois,
                                runoffsrc='srplusgwr',
                                correct_global_lakes=True,
                                srsmoothing=False,
                                l12harm=False,
                                discorr=True,
                                largerivercorr=True,
                                corrgridshift=True,
                                corrgridsmoothing=False,
                                thresholdperskm=0.001,
                                area_of_interest=aoi,
                                # corrweightfactor=0.1
                                )
    if rtype == 'ipg80' or rtype == 'clusterfull' or rtype == 'ipg50':
        down = DryverDownscalingWrapper(dconfig)
        down.prepare()
    if rtype == 'clustermod' or rtype == 'clusterfull':
        with open(os.path.join(dconfig.temp_dir, 'run_information.txt'), 'w') as f:
            temp = vars(dconfig)
            for item in temp:
                f.write('{} : {}\n'.format(item, temp[item]))
        dconfig.temp_dir = '/scratch/fuchydrosheds_/agdoell/trautmann/ddruns/'
        # dconfig.temp_dir = localdir
        dconfig.pickle(localdir)
    if rtype == 'ipg80':
        run_prepared_downscaling(localdir, 2)
    if rtype == 'ipg50':
        run_prepared_downscaling(localdir, 1)
        pass


if __name__ == '__main__':
    rootdir = os.path.dirname(os.path.abspath(
        getsourcefile(lambda:0))).split('\\src')[0]
    localdir = os.path.join(rootdir, 'results', 'localdir')
    if not os.path.exists(localdir):
        os.mkdir(localdir)
    main(rftype='ipg80',
         localdir=localdir,
         rootdir=rootdir)
    # run_prepared_downscaling(localdir, 2)
    # gather_finished_downscaling(localdir)


