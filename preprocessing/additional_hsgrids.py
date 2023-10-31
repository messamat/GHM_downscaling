import arcpy
from arcpy.sa import *
import os

arcpy.CheckOutExtension("Spatial")

def get_continental_hsgrids(continentlist, upapath, pixarea_path, globallakes_fraction_15s, outputdir,
                            overwrite, verbose=True):
    arcpy.env.workspace = r'in_memory'
    #xmin, xmax, ymin, ymax = get_continental_extent(continentlist)

    if verbose:
        print('Getting continental HydroSHEDS grids...')

    #Identify and write out raster of HR cell pourpoint for each LR cell
    cell_pourpoint_path = os.path.join(outputdir,
                                       '{}_cellpourpoint_15s.tif'.format(
                                           ''.join(continentlist)))
    if not arcpy.Exists(cell_pourpoint_path) or overwrite:
        upamax = Aggregate(in_raster=Raster(upapath), cell_factor=120, aggregation_type='MAXIMUM')
        arcpy.env.cellSize = 'MINOF'
        cellpourpoint = (Raster(upapath) == upamax)
        cellpourpoint.save(cell_pourpoint_path)
        del upamax, cellpourpoint

    #Clip HR pixel area and global lake fraction to the extent of the spatial union of continents in continentlist
    out_pixarea_clip = os.path.join(outputdir,'{}_pixarea_15s.tif'.format(''.join(continentlist)))
    if not arcpy.Exists(out_pixarea_clip) or overwrite:
        ExtractByMask(in_raster=pixarea_path,
                      in_mask_data=cell_pourpoint_path,
                      extraction_area='INSIDE').save(out_pixarea_clip)

    out_pixareafraction_glolakres_clip = os.path.join(outputdir,
                                                      '{}_pixareafraction_glolakres_15s.tif'.format(
                                                          ''.join(continentlist)))
    if not arcpy.Exists(out_pixareafraction_glolakres_clip) or overwrite:
        ExtractByMask(in_raster=globallakes_fraction_15s,
                      in_mask_data=cell_pourpoint_path,
                      extraction_area='INSIDE').save(out_pixareafraction_glolakres_clip)

