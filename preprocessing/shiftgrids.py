import os
import shutil
from open.DownstreamGrid import get_downstream_grid
import arcpy
from arcpy import env
from arcpy.sa import Raster, RasterCalculator, Con, IsNull, Aggregate, FlowAccumulation
arcpy.CheckOutExtension("Spatial")


def calc_shift_keep_largerivermask_grids(in_upstreamarea_path, in_flowdir_30min, in_area_flowacc_30min,
                                         tmpdir, outdir, continentlist):
    env.workspace = r'in_memory'
    #Compute maximum HR upstream drainage area in every 30 min cell
    outAggregate = Aggregate(in_raster=in_upstreamarea_path,
                             cell_factor=120,
                             aggregation_type='MAXIMUM')
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    arcpy.Clip_management(in_raster=Raster(in_flowdir_30min), out_raster='{}flowdir_30min.tif'.format(tmpdir),
                          in_template_dataset=outAggregate)
    arcpy.Clip_management(in_raster=Raster(in_area_flowacc_30min), out_raster='{}area_flowacc_30min.tif'.format(tmpdir),
                          in_template_dataset=outAggregate)
    outAggregate.save('{}MaxUpArea_05deg.tif'.format(tmpdir))

    #---------- Prepare shifting of correction terms to downstream HR grid cells ---------------------------------------
    get_downstream_grid(grid='{}MaxUpArea_05deg.tif'.format(tmpdir),
                        flowdir='{}flowdir_30min.tif'.format(tmpdir),
                        outgrid='{}MaxUpArea_down_05deg.tif'.format(tmpdir))
    MaxUpArea = Raster('{}MaxUpArea_05deg.tif'.format(tmpdir))
    MaxUpAreaDown = Raster('{}MaxUpArea_down_05deg.tif'.format(tmpdir))
    shift = RasterCalculator([MaxUpArea, MaxUpAreaDown], ["MaxUpArea", "MaxUpAreaDown"],
                             "Con((MaxUpAreaDown > 0.9 * MaxUpArea),((2 * MaxUpAreaDown) - MaxUpArea)/(2*MaxUpAreaDown), 0)")
    keep = 1 - shift
    shift.save('{}{}_shiftgrid.tif'.format(outdir, ''.join(continentlist)))
    keep.save('{}{}_keepgrid.tif'.format(outdir, ''.join(continentlist)))

    #---------- Prepare correction for differing characteristics of the river networks in rivers with large catchments -
    flowacc30min = Raster('{}area_flowacc_30min.tif'.format(tmpdir))
    # Difference between max HR upa in each LR cell from HydroSHEDS and upa in the LR cell from DDRM30
    areadifp = RasterCalculator([MaxUpArea, flowacc30min], ["MaxUpArea", "flowacc30min"],
                                "((MaxUpArea-flowacc30min)/flowacc30min) * 100")
    areadifp.save('{}areadifp.tif'.format(tmpdir))
    #Make a mask of large rivers for which upa for HydroSHEDS and DDRM30 match relatively well
    largeRiverMask = Con((((MaxUpArea >= 100000) & (areadifp >= -50) & (areadifp <= 50)) |
                          ((MaxUpArea >= 50000) & (areadifp >= -20) & (areadifp <= 20))), 1, 0)
    largeRiverMask = Con(IsNull(largeRiverMask), 0, largeRiverMask)
    largeRiverMask.save('{}{}_largeRiverMask.tif'.format(outdir, ''.join(continentlist)))
    gapMask = Con(largeRiverMask == 0, 1, 0) #opposite of largeRiverMask
    gapflowdir = Raster('{}flowdir_30min.tif'.format(tmpdir)) * gapMask
    gap_flowacc = FlowAccumulation(in_flow_direction_raster=gapflowdir, data_type="INTEGER")
    gap_flowacc.save('{}{}_gap_flowacc.tif'.format(outdir, ''.join(continentlist)))
    del MaxUpArea, MaxUpAreaDown, areadifp, outAggregate, gap_flowacc, flowacc30min, gapflowdir, gapMask, keep, shift, largeRiverMask
    shutil.rmtree(tmpdir)
