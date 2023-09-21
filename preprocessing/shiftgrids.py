import os
import shutil
from open.DownstreamGrid import get_downstream_grid
import arcpy
from arcpy import env
from arcpy.sa import Raster, RasterCalculator, Con, IsNull, Aggregate, FlowAccumulation
arcpy.CheckOutExtension("Spatial")


def calc_shift_keep_largerivers_mask_grids(in_upa_path, in_flowdir_30min, in_area_flowacc_30min,
                                         tmpdir, outdir, continentlist):
    env.workspace = r'in_memory'
    #Compute maximum HR upstream drainage area in every 30 min cell
    max_HRupa_30min = Aggregate(in_raster=in_upa_path,
                                cell_factor=120,
                                aggregation_type='MAXIMUM')
    if not os.path.exists(tmpdir):
        os.makedirs(tmpdir)

    arcpy.Clip_management(in_raster=Raster(in_flowdir_30min), 
                          out_raster=os.path.join(tmpdir, 'flowdir_30min.tif'),
                          in_template_dataset=max_HRupa_30min)
    arcpy.Clip_management(in_raster=Raster(in_area_flowacc_30min), 
                          out_raster=os.path.join(tmpdir, 'area_flowacc_30min.tif'),
                          in_template_dataset=max_HRupa_30min)
    max_HRupa_30min.save('{}max_HRupa_30min.tif'.format(tmpdir))

    #---------- Prepare shifting of correction terms to downstream HR grid cells ---------------------------------------
    get_downstream_grid(grid=os.path.join(tmpdir, 'max_HRupa_30min.tif'),
                        flowdir=os.path.join(tmpdir, 'flowdir_30min.tif'),
                        outgrid=os.path.join(tmpdir, 'max_HRupa_downstream_30min.tif')
                        )
    max_HRupa_30min = Raster('{}max_upa_30min.tif'.format(tmpdir))
    max_HRupa_downstream_30min = Raster('{}max_upa_down_30min.tif'.format(tmpdir))
    shift = Con((max_HRupa_downstream_30min > (0.9 * max_HRupa_30min)),
                ((2 * max_HRupa_downstream_30min) - max_HRupa_30min)/(2*max_HRupa_downstream_30min),
                0)
    keep = 1 - shift
    shift.save(os.path.join(outdir, '{}_shiftgrid.tif'.format(''.join(continentlist))))
    keep.save(os.path.join(outdir, '{}_keepgrid.tif'.format(''.join(continentlist))))

    #---------- Prepare correction for differing characteristics of the river networks in rivers with large catchments -
    flowacc30min = Raster(os.path.join(tmpdir, 'area_flowacc_30min.tif'))
    # Difference between max HR upa in each LR cell from HydroSHEDS and upa in the LR cell from DDRM30
    areadifp = (((max_HRupa_30min-flowacc30min)/flowacc30min) * 100)
    areadifp.save(os.path.join(tmpdir, 'areadifp.tif'))
    
    #Make a mask of large rivers for which upa for HydroSHEDS and DDRM30 match relatively well
    largerivers_mask = Con((((max_HRupa_30min >= 100000) & (areadifp >= -50) & (areadifp <= 50)) |
                          ((max_HRupa_30min >= 50000) & (areadifp >= -20) & (areadifp <= 20))),
                           1,
                           0)

    largerivers_mask = Con(IsNull(largerivers_mask), 0, largerivers_mask)
    largerivers_mask.save(os.path.join(outdir, '{}_largerivers_mask.tif'.format(''.join(continentlist))))

    gapMask = Con(largerivers_mask == 0, 1, 0) #opposite of largerivers_mask
    gapflowdir = Raster(os.path.join(tmpdir, 'flowdir_30min.tif')) * gapMask

    gap_flowacc = FlowAccumulation(in_flow_direction_raster=gapflowdir, data_type="INTEGER")
    gap_flowacc.save(os.path.join(outdir, '{}_gap_flowacc.tif'.format(''.join(continentlist))))

    del max_HRupa_30min, max_HRupa_downstream_30min, areadifp, gap_flowacc, flowacc30min, gapflowdir, (
        gapMask), keep, shift, largerivers_mask
    shutil.rmtree(tmpdir)
