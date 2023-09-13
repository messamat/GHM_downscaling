import arcpy
from arcpy.sa import FlowAccumulation, Raster
arcpy.CheckOutExtension("Spatial")


def flow_accumulation(outpathsfa, outpathup, flowdirp, pixareap):
    outFlowAccumulation = FlowAccumulation(in_flow_direction_raster=flowdirp, data_type="INTEGER")
    outFlowAccumulation.save(outpathsfa)
    arcpy.Clip_management(in_raster=pixareap, out_raster='in_memory/pixarea',
                          in_template_dataset=outFlowAccumulation)
    del outFlowAccumulation
    outFlowAccumulation = FlowAccumulation(in_flow_direction_raster=flowdirp, in_weight_raster=Raster('in_memory/pixarea'))
    (Raster('in_memory/pixarea') + outFlowAccumulation).save(outpathup)
