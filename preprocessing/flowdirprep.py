import arcpy


def combine_flow_direction_raster(hsflowdirgdbpath, continentlist , outpath):
    arcpy.env.workspace = hsflowdirgdbpath
    if len(continentlist) > 1:
        arcpy.MosaicToNewRaster_management(input_rasters=";".join(['{}_dir_15s'.format(x) for x in continentlist]),
                                           output_location=outpath,
                                           raster_dataset_name_with_extension=''.join(x for x in continentlist)
                                                                              + '_dir_15s.tif',
                                           number_of_bands=1)
    elif len(continentlist) == 1:
        arcpy.CopyRaster_management(in_raster='{}_dir_15s'.format(continentlist[0]),
                                    out_rasterdataset='{}{}_dir_15s.tif'.format(outpath, continentlist[0]))
    else:
        raise Exception("Please handover proper continentlist")
