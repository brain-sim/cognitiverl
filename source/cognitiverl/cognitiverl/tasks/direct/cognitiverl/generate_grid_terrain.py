#!/usr/bin/env python3

"""Script to view terrain - either from USD file or generate on the fly."""

import argparse
import os

# Launch Isaac Sim first before importing anything else
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(
    description="View terrain from USD file or generate on the fly"
)
parser.add_argument(
    "--terrain_path",
    type=str,
    default="/home/chandramouli/cognitiverl/source/cognitiverl/cognitiverl/tasks/direct/custom_assets/grid_terrain.usd",
    help="Path to terrain USD file",
)
parser.add_argument(
    "--generate",
    action="store_true",
    help="Generate terrain on the fly instead of loading from file",
)
parser.add_argument(
    "--save_terrain",
    type=str,
    default="/home/chandramouli/cognitiverl/source/cognitiverl/cognitiverl/tasks/direct/custom_assets/grid_terrain.usd",
    help="Path to save generated terrain as USD file",
)
parser.add_argument(
    "--terrain_size",
    type=float,
    nargs=2,
    default=[10.0, 10.0],
    help="Terrain size if generating (x, y)",
)
parser.add_argument(
    "--grid_size", type=float, default=0.45, help="Grid cell size for generated terrain"
)

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args_cli = parser.parse_args()

# Launch the simulator application (NOT headless so we can see it)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Now import everything else after Isaac Sim is launched."""

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import isaacsim.core.utils.prims as prim_utils
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.utils.viewports import set_camera_view


def save_terrain_as_usd(save_path: str):
    """Save the current terrain as USD file."""
    print(f"💾 Saving terrain to: {save_path}")

    try:
        import omni.usd
        from pxr import Usd

        # Get the current stage
        stage = omni.usd.get_context().get_stage()

        # Create a new layer for export
        export_layer = Usd.Stage.CreateNew(save_path)

        # Copy terrain prims to export stage, but save them at root level
        terrain_paths = ["/World/terrain", "/World/ground_plane", "/World/ground"]

        saved_terrain = None
        for terrain_path in terrain_paths:
            terrain_prim = stage.GetPrimAtPath(terrain_path)
            if terrain_prim.IsValid():
                print(f"📋 Copying prim: {terrain_path}")

                # Save at root level instead of with full path
                root_name = terrain_prim.GetName()
                root_path = f"/{root_name}"

                # Define the prim in the export stage at root level
                export_prim = export_layer.DefinePrim(
                    root_path, terrain_prim.GetTypeName()
                )

                # Copy all attributes
                for attr in terrain_prim.GetAttributes():
                    attr_name = attr.GetName()
                    attr_value = attr.Get()
                    if attr_value is not None:
                        try:
                            export_attr = export_prim.CreateAttribute(
                                attr_name, attr.GetTypeName()
                            )
                            export_attr.Set(attr_value)
                        except Exception as attr_e:
                            print(f"⚠️ Could not copy attribute {attr_name}: {attr_e}")

                # Copy children recursively
                def copy_children(source_prim, dest_stage, dest_path):
                    for child in source_prim.GetChildren():
                        child_name = child.GetName()
                        child_path = f"{dest_path}/{child_name}"
                        try:
                            dest_child = dest_stage.DefinePrim(
                                child_path, child.GetTypeName()
                            )

                            # Copy attributes
                            for attr in child.GetAttributes():
                                attr_name = attr.GetName()
                                attr_value = attr.Get()
                                if attr_value is not None:
                                    try:
                                        dest_attr = dest_child.CreateAttribute(
                                            attr_name, attr.GetTypeName()
                                        )
                                        dest_attr.Set(attr_value)
                                    except Exception as attr_e:
                                        print(
                                            f"⚠️ Could not copy child attribute {attr_name}: {attr_e}"
                                        )

                            # Recursively copy grandchildren
                            copy_children(child, dest_stage, child_path)
                        except Exception as child_e:
                            print(f"⚠️ Could not copy child {child_name}: {child_e}")

                copy_children(terrain_prim, export_layer, root_path)

                # Set this as the default primitive
                export_layer.SetDefaultPrim(export_prim)
                saved_terrain = root_path
                break  # Only copy the first valid terrain prim

        if saved_terrain:
            # Save the stage
            export_layer.GetRootLayer().Save()
            print(
                f"✅ Terrain saved successfully to: {save_path} (default prim: {saved_terrain})"
            )
            return True
        else:
            print("❌ No valid terrain found to save")
            return False

    except Exception as e:
        print(f"❌ Failed to save terrain: {e}")
        import traceback

        traceback.print_exc()
        return False


def generate_terrain():
    """Generate terrain on the fly."""
    print(f"🏗️ Generating terrain of size {args_cli.terrain_size}...")

    try:
        # Generate the grid terrain
        terrain_cfg = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=terrain_gen.TerrainGeneratorCfg(
                seed=1,
                use_cache=True,
                size=tuple(args_cli.terrain_size),  # Size of terrain
                num_rows=1,  # Single terrain patch
                num_cols=1,  # Single terrain patch
                sub_terrains={
                    "grid_terrain": terrain_gen.MeshRandomGridTerrainCfg(
                        proportion=1.0,
                        grid_width=0.45,
                        grid_height_range=(0.01, 0.06),
                        platform_width=3.0,
                    ),
                },
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.06, 0.08, 0.1),  # Dark blue-gray color
            ),
            max_init_terrain_level=0,  # Single level to avoid complexity
            debug_vis=False,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,  # Ensure >= 0.1
                dynamic_friction=1.0,  # Ensure >= 0.1
                restitution=0.0,
            ),
        )

        terrain_importer = TerrainImporter(terrain_cfg)
        print("✅ Grid terrain generated successfully!")
        return terrain_importer

    except Exception as e:
        print(f"❌ Grid terrain generation failed: {e}")
        return None


def load_terrain_from_usd(terrain_path: str):
    """Load terrain from USD file."""
    print(f"📁 Loading terrain from: {terrain_path}")


    try:
        # Method 1: Try direct USD spawning first
        terrain_cfg = sim_utils.UsdFileCfg(usd_path=terrain_path)
        terrain_cfg.func("/World/loaded_terrain", terrain_cfg)
        print("✅ Terrain loaded successfully from USD!")
        return True

    except Exception as e1:
        print(f"⚠️ Manual USD loading failed: {e1}")
        return None


def main():
    """Main function to view terrain."""

    # Initialize simulation context
    sim_params = {
        "use_gpu": True,
        "use_gpu_pipeline": False,
        "use_flatcache": False,
        "use_fabric": False,
        "enable_scene_query_support": True,
    }
    sim = SimulationContext(
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        sim_params=sim_params,
        backend="torch",
        device="cuda:0",
    )

    # Initialize simulation FIRST
    print("⚙️ Initializing simulation...")
    sim.reset()

    # Load or generate terrain
    terrain_loaded = False
    terrain_importer = None

    if args_cli.generate:
        # Generate terrain on the fly
        terrain_importer = generate_terrain()
        terrain_loaded = terrain_importer is not None

        # Save terrain if requested
        if terrain_loaded and args_cli.save_terrain:
            save_path = args_cli.save_terrain
            if not save_path.endswith(".usd"):
                save_path += ".usd"
            save_terrain_as_usd(save_path)
    else:
        # Try to load from USD file
        terrain_path = args_cli.terrain_path
        if not os.path.isabs(terrain_path):
            terrain_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), terrain_path)
            )

        print(f"🔍 Looking for terrain file at: {terrain_path}")
        if os.path.exists(terrain_path):
            result = load_terrain_from_usd(terrain_path)
            terrain_loaded = result is not None
            # Handle both TerrainImporter object and boolean returns
            if isinstance(result, TerrainImporter):
                terrain_importer = result
        else:
            print(f"❌ Terrain file not found at: {terrain_path}")
            print("🔄 Falling back to generating terrain...")
            terrain_importer = generate_terrain()
            terrain_loaded = terrain_importer is not None

    if not terrain_loaded:
        print("❌ Failed to load or generate terrain!")
        simulation_app.close()
        return

    # Add a single distant light
    print("💡 Adding distant light...")
    try:
        import omni.usd

        prim_utils.create_prim("/World/Sun", "DistantLight")
        stage = omni.usd.get_context().get_stage()
        sun_prim = stage.GetPrimAtPath("/World/Sun")
        if sun_prim.IsValid():
            # Set light properties
            sun_prim.GetAttribute("inputs:intensity").Set(3000.0)
            sun_prim.GetAttribute("inputs:color").Set((1.0, 1.0, 0.9))
            sun_prim.GetAttribute("inputs:angle").Set(5.0)
            print("✅ Distant light created")
    except Exception as e:
        print(f"⚠️ Could not create distant light: {e}")

    # Set up camera view
    if args_cli.generate:
        camera_distance = max(args_cli.terrain_size) * 2
        camera_height = max(args_cli.terrain_size) / 2
    else:
        camera_distance = 50.0
        camera_height = 25.0

    print(f"📷 Setting camera to distance: {camera_distance}, height: {camera_height}")
    set_camera_view([camera_distance, camera_distance, camera_height], [0.0, 0.0, 0.0])

    # Debug: Print what prims exist in the scene
    print("🔍 Checking scene contents...")
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    world_prim = stage.GetPrimAtPath("/World")
    if world_prim.IsValid():
        for child in world_prim.GetChildren():
            child_name = child.GetName()
            child_type = child.GetTypeName()
            print(f"   - Found prim: {child.GetPath()} (type: {child_type})")

            # Check for terrain-related prims
            if child_name.lower() in [
                "ground",
                "terrain",
                "ground_plane",
                "loaded_terrain",
            ]:
                print("     🌍 This looks like terrain!")
                # Check if it has geometry
                for grandchild in child.GetChildren():
                    print(
                        f"       - Child: {grandchild.GetPath()} (type: {grandchild.GetTypeName()})"
                    )

    # Start simulation
    if not sim.is_playing():
        sim.play()

    # Run the simulation
    print("\n🎮 Simulation running! Close the window to exit.")
    print("📷 Camera Controls:")
    print("   • Mouse drag: Rotate camera")
    print("   • Mouse wheel: Zoom in/out")
    print("   • Middle mouse drag: Pan camera")
    print("   • Alt + mouse: Alternative navigation")
    print("   • Try zooming out to see the full terrain")

    while simulation_app.is_running():
        # If simulation is stopped, then exit.
        if sim.is_stopped():
            break
        # If simulation is paused, then skip.
        if not sim.is_playing():
            sim.step()
            continue

        # Step simulation
        sim.step()

    # Stop simulation before closing
    sim.stop()

    # Close the simulation
    simulation_app.close()
    print("👋 Simulation closed!")


if __name__ == "__main__":
    main()
