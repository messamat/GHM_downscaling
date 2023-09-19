import arcpy
from arcpy import env
from arcpy.sa import *
arcpy.CheckOutExtension("Spatial")
from open.helper import get_continental_extent


def get_continental_hsgrids(continentlist, upareapath, in_pixarea_path, glolakpixareafr, outputdir):
    env.workspace = r'in_memory'
    xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
    #Identify and write out raster of HR cell pourpoint for each LR cell (but not guaranteed to perfectly align right?)
    upareamax = Aggregate(in_raster=Raster(upareapath), cell_factor=120, aggregation_type='MAXIMUM')
    env.cellSize = 'MINOF'
    cellpourpoint = (Raster(upareapath) == upareamax)
    cellpourpoint.save('{}{}_cellpourpoint_15s.tif'.format(outputdir, ''.join(continentlist)))
    del upareamax, cellpourpoint

    #Clip HR pixel area and global lake fraction to the extent of the spatial union of continents in continentlist
    arcpy.Clip_management(Raster(in_pixarea_path), "{} {} {} {}".format(xmin, ymin, xmax, ymax)).save(
        '{}{}_pixarea_15s.tif'.format(outputdir, ''.join(continentlist)))
    arcpy.Clip_management(Raster(glolakpixareafr), "{} {} {} {}".format(xmin, ymin, xmax, ymax)).save(
        '{}{}_pixareafraction_glolakres_15s.tif'.format(outputdir, ''.join(continentlist)))
