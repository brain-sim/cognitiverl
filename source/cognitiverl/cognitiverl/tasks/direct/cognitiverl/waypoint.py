import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg

##
# configuration
##

WAYPOINT_CFG = VisualizationMarkersCfg(
    prim_path="/World/Visuals/Cones",
    markers={
        "marker0": sim_utils.SphereCfg(  # future waypoint (yellow)
            radius=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
        "marker1": sim_utils.SphereCfg(  # current goal (green)
            radius=3.0,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 1.0, 0.0), opacity=0.9
            ),
        ),
        "marker2": sim_utils.SphereCfg(  # finished waypoint (white and invisible)
            radius=0.0,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.0, 0.0, 0.0), opacity=0.0
            ),
        ),
        "marker3": sim_utils.SphereCfg(  # avoid waypoint (red)
            radius=0.5,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
    },
)
