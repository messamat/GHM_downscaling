import arcpy
import os


def copygrids(rasters, inputdir, outputdir):
    for x in rasters:
        arcpy.management.CopyRaster(os.path.join(inputdir, x),
                                    os.path.join(outputdir, x))