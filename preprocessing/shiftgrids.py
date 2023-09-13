import os
import shutil
from open.DownstreamGrid import get_downstream_grid
import arcpy
from arcpy import env
from arcpy.sa import Raster, RasterCalculator, Con, IsNull, Aggregate, FlowAccumulation
arcpy.CheckOutExtension("Spatial")


def calc_shift_keep_largerivermask_grids(upstreampath, flowdir30min, areaflowacc30min, tmpdir, outdir, continentlist):
    env.workspace = r'in_memory'
    outAggregate = Aggregate(in_raster=upstreampath,
                             cell_factor=120,
                             aggregation_type='MAXIMUM')
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    arcpy.Clip_management(in_raster=Raster(flowdir30min), out_raster='{}flowdir30min.tif'.format(tmpdir),
                          in_template_dataset=outAggregate)
    arcpy.Clip_management(in_raster=Raster(areaflowacc30min), out_raster='{}areaflowacc30min.tif'.format(tmpdir),
                          in_template_dataset=outAggregate)
    outAggregate.save('{}MaxUpArea_05deg.tif'.format(tmpdir))
    get_downstream_grid('{}MaxUpArea_05deg.tif'.format(tmpdir),
                        '{}flowdir30min.tif'.format(tmpdir),
                        '{}MaxUpArea_down_05deg.tif'.format(tmpdir))
    MaxUpArea = Raster('{}MaxUpArea_05deg.tif'.format(tmpdir))
    MaxUpAreaDown = Raster('{}MaxUpArea_down_05deg.tif'.format(tmpdir))
    shift = RasterCalculator([MaxUpArea, MaxUpAreaDown], ["MaxUpArea", "MaxUpAreaDown"],
                             "Con((MaxUpAreaDown > 0.9 * MaxUpArea),((2 * MaxUpAreaDown) - MaxUpArea)/(2*MaxUpAreaDown), 0)")
    keep = 1 - shift
    shift.save('{}{}_shiftgrid.tif'.format(outdir, ''.join(continentlist)))
    keep.save('{}{}_keepgrid.tif'.format(outdir, ''.join(continentlist)))
    flowacc30min = Raster('{}areaflowacc30min.tif'.format(tmpdir))
    areadifp = RasterCalculator([MaxUpArea, flowacc30min], ["MaxUpArea", "flowacc30min"],
                                "((MaxUpArea-flowacc30min)/flowacc30min) * 100")
    areadifp.save('{}areadifp.tif'.format(tmpdir))
    largeRiverMask = Con((((MaxUpArea >= 100000) & (areadifp >= -50) & (areadifp <= 50)) |
                          ((MaxUpArea >= 50000) & (areadifp >= -20) & (areadifp <= 20))), 1, 0)
    largeRiverMask = Con(IsNull(largeRiverMask), 0, largeRiverMask)
    largeRiverMask.save('{}{}_largeRiverMask.tif'.format(outdir, ''.join(continentlist)))
    gapMask = Con(largeRiverMask == 0, 1, 0)
    gapflowdir = Raster('{}flowdir30min.tif'.format(tmpdir)) * gapMask
    gapfa = FlowAccumulation(in_flow_direction_raster=gapflowdir, data_type="INTEGER")
    gapfa.save('{}{}_gapfa.tif'.format(outdir, ''.join(continentlist)))
    del MaxUpArea, MaxUpAreaDown, areadifp, outAggregate, gapfa, flowacc30min, gapflowdir, gapMask, keep, shift, largeRiverMask
    shutil.rmtree(tmpdir)
