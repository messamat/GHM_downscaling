import arcpy

def combine_flow_direction_raster(hsflowdir_gdbpath, continentlist , outpath):
    arcpy.env.workspace = hsflowdir_gdbpath

    if len(continentlist) > 1:
        arcpy.management.MosaicToNewRaster(
            input_rasters=";".join(['{}_dir_15s'.format(x) for x in continentlist]),
            output_location=outpath,
            raster_dataset_name_with_extension='{}_dir_15s.tif'.format(''.join(x for x in continentlist)),
            number_of_bands=1)

    elif len(continentlist) == 1:
        arcpy.management.CopyRaster(in_raster='{}_dir_15s'.format(continentlist[0]),
                                    out_rasterdataset='{0}{1}_dir_15s.tif'.format(outpath, continentlist[0]))
    else:
        raise Exception("Please handover proper continentlist")
