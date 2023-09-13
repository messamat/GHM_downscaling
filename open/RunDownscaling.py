import pandas as pd
from codetiming import Timer

from DryverDownscalingWrapper import DryverDownscalingWrapper, gather_finished_downscaling, run_prepared_downscaling
from DryverDownscalingConfig import DownscalingConfig
from helper import get_continental_extent


@Timer(name='decorator', text='Downscaling takes currently {minutes:.0f} minutes')
def main(rtype):
    """
    Preparation and running script for Downscaling please modify variables in this main
    function to configure the downscaling and corrsponding paths.
    For description of configurationparameters please consult DryverDownscalingConfig class.

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
    wginpath = '/home/home1/gm/datasets/input_routing/wghm22e_v001/input/'
    wgpath = '/home/home8/dryver/22eant/'
    hspath = '/home/home1/gm/projects/DRYvER/03_data/12_downscalingdata_eu/'
    pois = pd.read_csv('{}stations.csv'.format(hspath))

    xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
    aoi = ((xmin, xmax), (ymin, ymax))
    if continent == 'rhone':
        aoi = ((3.5, 9), (43, 48.5))

    dconfig = DownscalingConfig(wg_in_path=wginpath,
                                wg_out_path=wgpath,
                                hs_path=hspath,
                                startyear=1901,
                                endyear=2019,
                                temp_dir=localdir,
                                write_raster=False,
                                write_result='nc',
                                mode='ts',
                                continent=continent,
                                pois=pois,
                                runoffsrc='srplusgwr',
                                glolakredist=True,
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
        with open(dconfig.temp_dir + 'run_information.txt', 'w') as f:
            temp = vars(dconfig)
            for item in temp:
                f.write('{} : {}\n'.format(item, temp[item]))
        dconfig.temp_dir = '/scratch/fuchs/agdoell/trautmann/ddruns/'
        # dconfig.temp_dir = localdir
        dconfig.pickle(localdir)
    if rtype == 'ipg80':
        run_prepared_downscaling(localdir, 2)
    if rtype == 'ipg50':
        run_prepared_downscaling(localdir, 1)
        pass


if __name__ == '__main__':
    localdir = '/home/home8/dryver/rastertemp/nobackup/eurasia/final/'
    main('ipg80')
    # run_prepared_downscaling(localdir, 2)
    # gather_finished_downscaling(localdir)


