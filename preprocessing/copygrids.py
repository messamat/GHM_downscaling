import arcpy


def copygrids(rasters, inputdir, outputdir):
    for x in rasters:
        arcpy.CopyRaster_management('{}{}'.format(inputdir, x), '{}{}'.format(outputdir, x))