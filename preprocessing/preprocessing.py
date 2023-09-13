
from flowdirprep import combine_flow_direction_raster
from simpleflowaccumulation import flow_accumulation
from shiftgrids import calc_shift_keep_largerivermask_grids
from additional_hsgrids import get_continental_hsgrids
from copygrids import copygrids
from stationsrastermapping import create_stationrastermapping


class PreProcessing:
    """
    Class to produce necessary grid files for Downscaling runoff into streamflow.

    As input following files (gdbs, raster) are necessary in the setup_folder:

    - flow_dir_15s_by_continent.gdb (with raster datasets containing flow directions of HydroSheds),
    - pixel_area_skm_15s.gdb\px_area_skm_15s (gdb raster dataset with area of HydroSheds grid cells),
    - flowdir_30min.tif (rasterfile with flow directions of WaterGAP (DDM30)),
    - orgDDM30area.tif (rasterfile with areas of WaterGAP grid cells),
    - pixareafraction_glolakres_15s.tif (rasterfile with values in HR grid cells which are covered by a global lake or
      reservoir, the value correspond to the area fraction of a HR grid cell to all global lakes or reservoirs in one
      LR grid cell )

    Following files / grids are produced in outputdir:

    - continents_dir_15s.tif (HR raster with flow directions clipped to continents of interest),
    - continents_shiftgrid.tif (LR raster with percentage of how much a correction value should be shifted downstream),
    - continents_keepgrid.tif (LR raster shiftgrid -1 i.e. how much a correction value should be kept),
    - continents_largeRiverMask.tif (LR raster with 1, where a *large* river with not mismatching LR and HR river networks
      is and 0 where not),
    - continents_gapfa.tif (LR flow accumulation grid of cells of in non *large* rivers with fitting river networks
      (largeRiverMask.Not)),
    - continents_cellpourpoint_15s.tif (HR grid with 1 where the HR grid cell has the maximum upstream area per LR grid
      cell),
    - continents_pixarea_15s.tif (HR grid with area of HR grid cells),
    - continents_pixareafraction_glolakres_15s.tif (rasterfile with values in HR grid cells which are covered by a
      global lake or reservoir, the value correspond to the area fraction of a HR grid cell to all global lakes or
      reservoirs in one LR grid cell),
    - landratio_correction.tif (LR grid on the land to continental area ratio to correct runoff values, is just copied)



    """
    def __init__(self, setup_folder, outputdir, tmpdir, continentlist):
        flowdirgdbpath = r'{}flow_dir_15s_by_continent.gdb'.format(setup_folder)
        pixelareapath = r'{}pixel_area_skm_15s.gdb\px_area_skm_15s'.format(setup_folder)
        flowdir05deg = r'{}flowdir_30min.tif'.format(setup_folder)
        areaflowacc05deg = r"{}orgDDM30area.tif".format(setup_folder)
        glolakredist = r'{}pixareafraction_glolakres_15s.tif'.format(setup_folder)
        landratio = r'landratio_correction.tif'
        stations = r'{}stations_europe.geojson'.format(setup_folder)

        flowdirpath = outputdir + ''.join(x for x in continentlist) + '_dir_15s.tif'
        flowaccpath = outputdir + ''.join(x for x in continentlist) + '_acc_15s.tif'
        upstreampath = outputdir + ''.join(x for x in continentlist) + '_upa_15s.tif'
        combine_flow_direction_raster(flowdirgdbpath, continentlist, outputdir)
        flow_accumulation(flowaccpath, upstreampath, flowdirpath, pixelareapath)
        calc_shift_keep_largerivermask_grids(upstreampath, flowdir05deg, areaflowacc05deg,
                                             tmpdir, outputdir, continentlist)
        get_continental_hsgrids(continentlist, upstreampath, pixelareapath, glolakredist, outputdir)
        copygrids([landratio], setup_folder, outputdir)
        create_stationrastermapping(flowaccpath, stations, outputdir)


if __name__ == '__main__':
    setup_folder = r'D:\\data_temp\\setupdata_for_downscaling\\'
    outputdir = r'D:\\data_temp\\hs_reproduced\\'
    tmpdir = r'D:\\data_temp\\hs_reproduced\\tmp\\'
    continentlist = ['eu', 'as', 'si']
    PreProcessing(setup_folder, outputdir, tmpdir, continentlist)
