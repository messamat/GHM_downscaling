# DRYvER

## Start by pre-processing
### Get input data
The following input files (gdbs, raster) are necessary in the setup_folder:

- flow_dir_15s_by_continent.gdb (with raster datasets containing flow directions of HydroSheds),
- pixel_area_skm_15s.gdb\px_area_skm_15s (gdb raster dataset with area of HydroSheds grid cells),
- flowdir_30min.tif (rasterfile with flow directions of WaterGAP (DDM30)),
- orgDDM30area.tif (rasterfile with areas of WaterGAP grid cells),
- pixareafraction_glolakres_15s.tif (rasterfile with values in HR grid cells which are covered by a global lake or
  reservoir, the value correspond to the area fraction of a HR grid cell to all global lakes or reservoirs in one
  LR grid cell )

"\bin\GDAL-3.4.3-cp39-cp39-win_amd64.whl" from https://www.lfd.uci.edu/~gohlke/pythonlibs/#gdal
