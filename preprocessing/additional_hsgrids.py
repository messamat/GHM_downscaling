import arcpy
from arcpy import env
from arcpy.sa import *
arcpy.CheckOutExtension("Spatial")
from open.helper import get_continental_extent


def get_continental_hsgrids(continentlist, upareapath, pixareapath, glolakpixareafr, outputdir):
    env.workspace = r'in_memory'
    xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
    upareamax = Aggregate(in_raster=Raster(upareapath), cell_factor=120, aggregation_type='MAXIMUM')
    env.cellSize = 'MINOF'
    cellpourpoint = (Raster(upareapath) == upareamax)
    cellpourpoint.save('{}{}_cellpourpoint_15s.tif'.format(outputdir, ''.join(continentlist)))
    del upareamax, cellpourpoint
    arcpy.Clip_management(Raster(pixareapath), "{} {} {} {}".format(xmin, ymin, xmax, ymax), "pixareacont")
    Raster("pixareacont").save('{}{}_pixarea_15s.tif'.format(outputdir, ''.join(continentlist)))
    arcpy.Clip_management(Raster(glolakpixareafr), "{} {} {} {}".format(xmin, ymin, xmax, ymax),
                          "pixareafraction_glolakres_15s")
    Raster("pixareafraction_glolakres_15s").save(
        '{}{}_pixareafraction_glolakres_15s.tif'.format(outputdir, ''.join(continentlist)))
