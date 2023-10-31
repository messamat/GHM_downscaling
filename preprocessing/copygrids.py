import arcpy
import os


def copygrids(rasters, inputdir, outputdir, overwrite):
    for x in rasters:
        outpath = os.path.join(outputdir, x)
        if not arcpy.Exists(outpath) or overwrite:
            arcpy.management.CopyRaster(os.path.join(inputdir, x),
                                        outpath)