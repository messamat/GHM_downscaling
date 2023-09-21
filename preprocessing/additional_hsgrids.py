import arcpy
from arcpy.sa import *
import os
from preprocessing.helper import get_continental_extent

arcpy.CheckOutExtension("Spatial")

def get_continental_hsgrids(continentlist, upapath, in_pixarea_path, glolakpixareafr, outputdir):
    arcpy.env.workspace = r'in_memory'
    xmin, xmax, ymin, ymax = get_continental_extent(continentlist)
    #Identify and write out raster of HR cell pourpoint for each LR cell (but not guaranteed to perfectly align right?)
    upamax = Aggregate(in_raster=Raster(upapath), cell_factor=120, aggregation_type='MAXIMUM')
    arcpy.env.cellSize = 'MINOF'
    cellpourpoint = (Raster(upapath) == upamax)
    cellpourpoint.save('{}{}_cellpourpoint_15s.tif'.format(outputdir, ''.join(continentlist)))
    del upamax, cellpourpoint

    #Clip HR pixel area and global lake fraction to the extent of the spatial union of continents in continentlist
    arcpy.management.Clip(Raster(in_pixarea_path), "{} {} {} {}".format(xmin, ymin, xmax, ymax)
                          ).save(os.path.join(outputdir,
                                              '{}_pixarea_15s.tif'.format(''.join(continentlist))
                                              )
    )
    arcpy.management.Clip(Raster(glolakpixareafr), "{} {} {} {}".format(xmin, ymin, xmax, ymax)
                          ).save(os.path.join(outputdir,
                                              '{}_pixareafraction_glolakres_15s.tif'.format(''.join(continentlist))
                                              )
    )
