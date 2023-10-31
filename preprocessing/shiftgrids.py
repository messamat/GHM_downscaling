import os
import shutil
from DownstreamGrid import get_downstream_grid
import arcpy
from arcpy import env
from arcpy.sa import Raster, RasterCalculator, Con, IsNull, Aggregate, FlowAccumulation
arcpy.CheckOutExtension("Spatial")


def calc_shift_keep_largerivers_mask_grids(in_upa_path, in_flowdir_30min, in_area_flowacc_30min,
                                           tmpdir, outdir, continentlist, overwrite,
                                           del_tmpdir=False, verbose=True):
    if verbose:
        print("Computing shift and keep grids and large rivers mask...")
    env.workspace = r'in_memory'

    #Compute maximum HR upstream drainage area in every 30 min cell
    max_HRupa_30min_path = os.path.join(tmpdir, 'max_HRupa_30min.tif')

    if not arcpy.Exists(max_HRupa_30min_path) or overwrite:
        max_HRupa_30min = Aggregate(in_raster=in_upa_path,
                                    cell_factor=120,
                                    aggregation_type='MAXIMUM')
        max_HRupa_30min.save(max_HRupa_30min_path)
    else:
        max_HRupa_30min = max_HRupa_30min_path

    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    flowdir_30min_clip = os.path.join(tmpdir, 'flowdir_30min.tif')
    if not arcpy.Exists(flowdir_30min_clip) or overwrite:
        arcpy.management.Clip(in_raster=Raster(in_flowdir_30min),
                              in_template_dataset=max_HRupa_30min,
                              out_raster=flowdir_30min_clip,
                              )

    flowacc_30min_clip = os.path.join(tmpdir, 'area_flowacc_30min.tif')
    if not arcpy.Exists(flowacc_30min_clip) or overwrite:
        arcpy.management.Clip(in_raster=Raster(in_area_flowacc_30min),
                              out_raster=flowacc_30min_clip,
                              in_template_dataset=max_HRupa_30min)


    #---------- Prepare shifting of correction terms to downstream HR grid cells ---------------------------------------
    max_HRupa_downstream_30min = os.path.join(tmpdir, 'max_HRupa_downstream_30min.tif')
    if not arcpy.Exists(max_HRupa_downstream_30min) or overwrite:
        get_downstream_grid(in_valuegrid=max_HRupa_30min_path,
                            in_flowdir=flowdir_30min_clip,
                            out_grid=max_HRupa_downstream_30min
                            )

    shift_grid = os.path.join(outdir, '{}_shiftgrid.tif'.format(''.join(continentlist)))
    keep_grid = os.path.join(outdir, '{}_keepgrid.tif'.format(''.join(continentlist)))

    if not arcpy.Exists(shift_grid) or overwrite:
        shift = Con(in_conditional_raster=(Raster(max_HRupa_downstream_30min) > (0.9 * Raster(max_HRupa_30min))),
                    in_true_raster_or_constant=(((2 * Raster(max_HRupa_downstream_30min)) - Raster(max_HRupa_30min))
                                                /(2*Raster(max_HRupa_downstream_30min))),
                    in_false_raster_or_constant=0)
        keep = 1 - shift
        shift.save(shift_grid)
        keep.save(keep_grid)

        del keep, shift

    #---------- Prepare correction for differing characteristics of the river networks in rivers with large catchments -
    largerivers_mask_path = os.path.join(outdir, '{}_largerivers_mask.tif'.format(''.join(continentlist)))

    if not arcpy.Exists(largerivers_mask_path) or overwrite:
        flowacc30min = Raster(flowacc_30min_clip)
        # Difference between max HR upa in each LR cell from HydroSHEDS and upa in the LR cell from DDRM30
        areadifp = (((max_HRupa_30min-flowacc30min)/flowacc30min) * 100)
        areadifp.save(os.path.join(tmpdir, 'areadifp.tif'))

        #Make a mask of large rivers for which upa for HydroSHEDS and DDRM30 match relatively well
        largerivers_mask = Con((((Raster(max_HRupa_30min) >= 100000) & (areadifp >= -50) & (areadifp <= 50)) |
                              ((Raster(max_HRupa_30min) >= 50000) & (areadifp >= -20) & (areadifp <= 20))),
                               1,
                               0)

        largerivers_mask = Con(IsNull(largerivers_mask), 0, largerivers_mask)
        largerivers_mask.save(largerivers_mask_path)

        gapMask = Con(largerivers_mask == 0, 1, 0) #opposite of largerivers_mask
        gapflowdir = Raster(flowdir_30min_clip) * gapMask

        gap_flowacc = FlowAccumulation(in_flow_direction_raster=gapflowdir, data_type="INTEGER")
        gap_flowacc.save(os.path.join(outdir, '{}_gap_flowacc.tif'.format(''.join(continentlist))))

        del areadifp, gap_flowacc, flowacc30min, gapflowdir, gapMask, largerivers_mask

    del max_HRupa_30min, max_HRupa_downstream_30min
    if del_tmpdir:
        shutil.rmtree(tmpdir)
