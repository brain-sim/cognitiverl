#!/usr/bin/env python3

"""Script to generate and save large terrain as USD file for faster loading."""

import argparse

# Launch Isaac Sim first before importing anything else
from isaaclab.app import AppLauncher

# Add command line arguments
parser = argparse.ArgumentParser(
    description="Generate large grid terrain and save as USD"
)
parser.add_argument(
    "--save_path",
    type=str,
    default="/home/chandramouli/cognitiverl/source/cognitiverl/cognitiverl/tasks/direct/custom_assets/large_grid_terrain.usd",
    help="Path to save terrain USD file",
)
parser.add_argument(
    "--terrain_size",
    type=float,
    nargs=2,
    default=[10.0, 10.0],
    help="Terrain size (x, y)",
)

# Append AppLauncher CLI args
AppLauncher.add_app_launcher_args(parser)

# Parse arguments
args_cli = parser.parse_args()

# Launch the simulator application
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Now import everything else after Isaac Sim is launched."""

import os
import time

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
import omni.kit.commands

# Import USD and Omniverse modules
import omni.usd
from isaaclab.terrains.terrain_importer import TerrainImporter
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg

# Import simulation context
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Usd


def verify_file_saved(file_path: str) -> bool:
    """Verify that the file was actually saved and get its info."""
    if os.path.exists(file_path):
        file_size = os.path.getsize(file_path)
        print(f"✅ File verified: {file_path}")
        print(f"📊 File size: {file_size / (1024 * 1024):.2f} MB")
        return True
    else:
        print(f"❌ File NOT found: {file_path}")
        return False


def save_terrain_usd(output_path: str) -> bool:
    """Try multiple methods to save the terrain as USD."""

    print(f"Saving terrain to {output_path}...")

    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"📁 Directory ensured: {os.path.dirname(output_path)}")

    # Method 1: Direct USD Stage Export
    try:
        stage = omni.usd.get_context().get_stage()
        if stage:
            print("🔍 Checking for terrain prim...")
            terrain_prim = stage.GetPrimAtPath("/World/ground")
            if terrain_prim.IsValid():
                print("✅ Terrain prim found, exporting stage...")
                result = stage.Export(output_path)
                print(f"📝 Export result: {result}")

                # Verify the file was created
                if verify_file_saved(output_path):
                    return True
                else:
                    print("⚠️ Export claimed success but file not found")
            else:
                print("❌ Terrain prim not found at /World/ground")
                # List all prims to see what's available
                print("🔍 Available prims:")
                for prim in stage.Traverse():
                    print(f"  - {prim.GetPath()}")
    except Exception as e:
        print(f"❌ Method 1 (USD Export) failed: {e}")

    # Method 2: Export specific terrain prim
    try:
        stage = omni.usd.get_context().get_stage()
        terrain_prim = stage.GetPrimAtPath("/World/ground")

        if terrain_prim.IsValid():
            print("🔄 Trying prim-specific export...")
            # Create a new layer and copy the terrain
            new_layer = Usd.Layer.CreateNew(output_path)
            edit_context = Usd.EditContext(stage, new_layer)

            with edit_context:
                # Copy the terrain prim to the new layer
                stage.DefinePrim("/World")
                Usd.Utils.CopyLayerToLayer(stage.GetRootLayer(), new_layer)

            new_layer.Save()

            if verify_file_saved(output_path):
                return True

    except Exception as e:
        print(f"❌ Method 2 (Prim Export) failed: {e}")

    # Method 3: Simple layer export
    try:
        stage = omni.usd.get_context().get_stage()
        if stage:
            print("🔄 Trying layer export...")
            root_layer = stage.GetRootLayer()
            success = root_layer.Export(output_path)
            print(f"📝 Layer export result: {success}")

            if verify_file_saved(output_path):
                return True

    except Exception as e:
        print(f"❌ Method 3 (Layer Export) failed: {e}")

    print("❌ All save methods failed!")
    return False


def main():
    """Generate and save terrain as USD file."""

    # Get the correct output path - make sure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, args_cli.save_path)

    print(f"📍 Script directory: {script_dir}")
    print(f"📁 Output path: {output_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📂 Created directory: {output_dir}")

    print(f"🌍 Generating large terrain of size {args_cli.terrain_size}...")

    # Initialize simulation context
    sim_params = {
        "use_gpu": True,
        "use_gpu_pipeline": True,
        "use_flatcache": True,
        "use_fabric": True,
        "enable_scene_query_support": True,
    }
    sim = SimulationContext(
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        sim_params=sim_params,
        backend="torch",
        device="cuda:0",
    )

    # Set camera view for better visualization (optional)
    set_camera_view([0.0, 30.0, 25.0], [0.0, 0.0, -2.5])

    # Create terrain configuration
    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            seed=1,
            use_cache=True,
            size=tuple(args_cli.terrain_size),  # Large terrain size
            num_rows=1,  # Single terrain patch
            num_cols=1,  # Single terrain patch
            sub_terrains={
                "grid_terrain": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=1.0,
                    grid_width=0.45,  # Width of each grid cell (45cm)
                    grid_height_range=(0.01, 0.06),  # Height variation (5-15cm)
                    platform_width=1.0,  # Size of flat platform at center
                ),
            },
        ),
        max_init_terrain_level=0,
        debug_vis=False,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.06, 0.08, 0.10),  # Dark blue-gray color
            emissive_color=(0.0, 0.0, 0.0),  # No emission
            roughness=0.9,  # High roughness for matte finish
            metallic=0.0,  # Non-metallic surface
            opacity=1.0,  # Fully opaque
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=2.0,  # High friction for terrain
            dynamic_friction=2.0,  # High friction for terrain
            restitution=0.0,
        ),
    )

    # Create the terrain importer
    print("🏗️ Creating terrain...")
    terrain_importer = TerrainImporter(terrain_cfg)

    # Reset and start simulation
    sim.reset()

    # Wait for terrain generation to complete
    print("⏳ Waiting for terrain generation...")

    # Run simulation steps to ensure terrain is fully generated
    for i in range(30):
        sim.step()
        time.sleep(0.1)
        if i % 10 == 0:
            print(f"   Step {i + 1}/30...")

    # Additional wait time
    time.sleep(5.0)
    print("✅ Terrain generation completed!")

    # Try to save the terrain using multiple methods
    success = save_terrain_usd(output_path)

    if success:
        print("\n🎉 Terrain generation and saving completed successfully!")
        print(f"📁 File saved at: {output_path}")
        print(
            f"📏 Terrain size: {args_cli.terrain_size[0]}x{args_cli.terrain_size[1]} units"
        )

        # Final verification
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"✅ Final verification: File exists ({file_size:,} bytes)")
        else:
            print("❌ Final verification: File does not exist!")

    else:
        print("\n❌ Failed to save terrain to USD file!")
        print(f"💡 Check permissions for directory: {os.path.dirname(output_path)}")

    # Stop simulation before closing
    sim.stop()

    # Close the simulation
    simulation_app.close()


if __name__ == "__main__":
    main()
