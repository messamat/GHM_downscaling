import os

from preprocessing.flowdirprep import combine_flow_direction_raster
from preprocessing.simpleflowaccumulation import flow_accumulation
from preprocessing.shiftgrids import calc_shift_keep_largerivers_mask_grids
from preprocessing.additional_hsgrids import get_continental_hsgrids
from preprocessing.copygrids import copygrids
from preprocessing.stationsrastermapping import create_stationrastermapping


class PreProcessing:
    """
    Class to produce necessary grid files for downscaling runoff into streamflow.

    The following input files (gdbs, raster) are necessary in the setup_folder:

    - flow_dir_15s_by_continent.gdb (with raster datasets containing flow directions of HydroSheds),
    - pixel_area_skm_15s.gdb\px_area_skm_15s (gdb raster dataset with area of HydroSheds grid cells),
    - flowdir_30min.tif (rasterfile with flow directions of WaterGAP (DDM30)),
    - orgDDM30area.tif (rasterfile with areas of WaterGAP grid cells),
    - pixareafraction_glolakres_15s.tif (rasterfile with values in HR grid cells which are covered by a global lake or
      reservoir, the value correspond to the fraction of the area of all global lakes or reservoirs in the LR grid cell
      contained in the HR grid cell. For example, if an LR grid cell is completely contained within a lake or reservoir,
      then allHR grid cells within would have a value of 1/14400)

    The following files / grids are produced in outputdir:

    - continents_dir_15s.tif (HR raster with flow directions clipped to continents of interest),
    - continents_shiftgrid.tif (LR raster with percentage of how much a correction value should be shifted downstream),
    - continents_keepgrid.tif (LR raster shiftgrid -1 i.e. how much a correction value should be kept),
    - continents_largerivers_mask.tif (LR raster with 1 where there is a *large* river with matching LR and HR river networks,
        and 0 in smaller rivers or large rivers with mismatches),
    - continents_gap_flowacc.tif (LR flow accumulation grid of cells which are not *large* rivers with fitting river networks
      (largerivers_mask.Not)),
    - continents_cellpourpoint_15s.tif (HR grid with 1 where the HR grid cell has the maximum upstream area per LR grid
      cell),
    - continents_pixarea_15s.tif (HR grid with area of HR grid cells),
    - pixareafraction_glolakres_15s.tif (rasterfile with values in HR grid cells which are covered by a global lake or
      reservoir, the value correspond to the fraction of the area of all global lakes or reservoirs in the LR grid cell
      contained in the HR grid cell. For example, if an LR grid cell is completely contained within a lake or reservoir,
      then allHR grid cells within would have a value of 1/14400)
    - landratio_correction.tif (LR grid on the land to continental area ratio to correct runoff values, is just copied)



    """
    def __init__(self, setup_folder, outputdir, tmpdir, continentlist):
        flowdir_gdbpath = r'{}flow_dir_15s_by_continent.gdb'.format(setup_folder)
        pixelarea_path = r'{}pixel_area_skm_15s.gdb\px_area_skm_15s'.format(setup_folder)
        flowdir_30min = r'{}flowdir_30min.tif'.format(setup_folder)
        area_flowacc_30min = r"{}orgDDM30area.tif".format(setup_folder)
        globallakes_fraction_15s = r'{}pixareafraction_glolakres_15s.tif'.format(setup_folder)
        landratio = r'landratio_correction.tif'
        stations = r'{}stations_europe.geojson'.format(setup_folder)

        flowdir_path = os.path.join(outputdir,
                                    '{}_dir_15s.tif'.format(''.join(x for x in continentlist)))
        flowacc_path = os.path.join(outputdir,
                                    '{}_acc_15s.tif'.format(''.join(x for x in continentlist)))
        upa_path = os.path.join(outputdir,
                                    '{}_upa_15s.tif'.format(''.join(x for x in continentlist)))

        # Standard mosaic flow dir rasters
        combine_flow_direction_raster(flowdir_gdbpath, continentlist, outputdir)

        # Standard flow acc -> upa_path
        flow_accumulation(in_flowdir_path=flowdir_path,
                          in_pixarea_path=pixelarea_path,
                          out_flowacc_path=flowacc_path,
                          out_upa_path=upa_path
                          )

        # Prepare shifting of correction terms to downstream HR grid cells
        # and correction for differing characteristics of the river networks in rivers with large catchments
        calc_shift_keep_largerivers_mask_grids(upa_path, flowdir_30min, area_flowacc_30min,
                                               tmpdir, outputdir, continentlist)

        # compute HR pourpoints of LR cells and clip HR pixel area and HR global lake fractions to
        # union of continents in continentlist
        get_continental_hsgrids(continentlist, upa_path, pixelarea_path, globallakes_fraction_15s, outputdir)

        #Copy 'landratio_correction.tif' to setup folder
        copygrids([landratio], setup_folder, outputdir)

        #Get cell position (row and column indices) of stations in HR flow accumulation raster
        create_stationrastermapping(flowacc_path, stations, outputdir)


if __name__ == '__main__':
    import os
    from inspect import getsourcefile

    # Folder structure
    rootdir = (os.path.dirname(os.path.abspath(
            getsourcefile(lambda: 0)))).split('\\src')[0]
    datdir = os.path.join(rootdir, 'data')
    resdir = os.path.join(rootdir, 'results')

    setup_folder = os.path.join(datdir, "setupdata_for_downscaling")

    outputdir = os.path.join(datdir, "hs_reproduced")
    tmpdir = os.path.join(datdir, "hs_reproduced", "tmp")
    continentlist = ['eu', 'as', 'si']

    PreProcessing(setup_folder, outputdir, tmpdir, continentlist)
