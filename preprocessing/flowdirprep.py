import arcpy
import os

def combine_flow_direction_raster(hydrosheds_flowdirgdb_path, continentlist, outputdir,
                                  verbose=True, overwrite=True):
    arcpy.env.workspace = hydrosheds_flowdirgdb_path

    outpath = os.path.join(outputdir, '{}_dir_15s.tif').format(''.join(x for x in continentlist))

    if not arcpy.Exists(outpath) or overwrite:
        if verbose:
            print("Combining flow direction rasters...")

        if len(continentlist) > 1:
            if verbose:
                print("Multiple rasters were provided. Mosaicking them...")
            arcpy.management.MosaicToNewRaster(
                input_rasters=";".join(['{}_dir_15s'.format(x) for x in continentlist]),
                output_location=outputdir,
                raster_dataset_name_with_extension=os.path.split(outpath)[1],
                number_of_bands=1)

        elif len(continentlist) == 1:
            if verbose:
                print("A single raster was provided. Copying it...")
            arcpy.management.CopyRaster(in_raster='{}_dir_15s'.format(continentlist[0]),
                                        out_rasterdataset=outpath)
        else:
            raise Exception("Please handover proper continentlist")
    else:
        if verbose:
            print("{} already exists and overwrite==False. Skipping...".format(outpath))

