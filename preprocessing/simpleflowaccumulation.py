import arcpy
from arcpy.sa import FlowAccumulation, Raster
arcpy.CheckOutExtension("Spatial")


def flow_accumulation(in_flowdir_path, in_pixarea_path, out_flowacc_path, out_upa_path):
    outFlowAccumulation = FlowAccumulation(in_flow_direction_raster=in_flowdir_path, data_type="INTEGER")
    outFlowAccumulation.save(out_flowacc_path)
    arcpy.management.Clip(in_raster=in_pixarea_path, out_raster='in_memory/pixarea',
                          in_template_dataset=outFlowAccumulation)
    del outFlowAccumulation
    outFlowAccumulation = FlowAccumulation(in_flow_direction_raster=in_flowdir_path,
                                           in_weight_raster=Raster('in_memory/pixarea'))
    (Raster('in_memory/pixarea') + outFlowAccumulation).save(out_upa_path)
