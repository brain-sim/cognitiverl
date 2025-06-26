#!/usr/bin/env python3

"""Script to generate and save large terrain as USD file for faster loading."""

import argparse
import asyncio
import os


def main():
    """Generate and save terrain as USD file."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate large grid terrain and save as USD"
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    args = parser.parse_args()

    # Import Isaac Lab modules after argument parsing
    from isaaclab.app import AppLauncher

    # Launch Isaac Sim
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    # Now import the rest after Isaac Sim is launched
    import time

    import isaaclab.sim as sim_utils
    import isaaclab.terrains as terrain_gen
    import omni.usd
    from isaaclab.terrains import TerrainImporter, TerrainImporterCfg

    # Configuration
    terrain_size = (4096 * 40.0, 4096 * 40.0)  # Large terrain size
    output_path = os.path.join(
        os.path.dirname(__file__), "custom_assets", "large_grid_terrain.usd"
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Generating large terrain of size {terrain_size}...")

    # Create terrain configuration
    terrain_cfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=terrain_gen.TerrainGeneratorCfg(
            seed=1,
            use_cache=True,
            size=terrain_size,  # Large terrain size
            border_width=1.0,
            num_rows=1,  # Single terrain patch
            num_cols=1,  # Single terrain patch
            horizontal_scale=0.5,
            vertical_scale=0.01,
            slope_threshold=0.75,
            sub_terrains={
                "grid_terrain": terrain_gen.MeshRandomGridTerrainCfg(
                    proportion=1.0,
                    grid_width=0.45,  # Width of each grid cell (45cm)
                    grid_height_range=(0.05, 0.15),  # Height variation (5-15cm)
                    platform_width=1.0,  # Size of flat platform at center
                    border_width=1.0,  # Match outer border width
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
    print("Creating terrain...")
    terrain_importer = TerrainImporter(terrain_cfg)

    # Wait for terrain generation to complete
    print("Waiting for terrain generation...")
    time.sleep(5.0)

    # Save the stage as USD
    print(f"Saving terrain to {output_path}...")

    async def save_terrain():
        """Async function to save the terrain."""
        try:
            stage = omni.usd.get_context().get_stage()
            result = await omni.usd.get_context().save_as_stage_async(output_path)
            print(f"Save result: {result}")
            return result
        except Exception as e:
            print(f"Error saving terrain: {e}")
            return False

    # Create event loop and run save operation

    async def run_save():
        """Run the save operation and close simulation."""
        success = await save_terrain()
        if success:
            print(f"Terrain generation complete! Saved to: {output_path}")
        else:
            print("Failed to save terrain!")

        # Close the simulation
        simulation_app.close()

    # Run the async operation
    try:
        asyncio.run(run_save())
    except Exception as e:
        print(f"Error in async operation: {e}")
        simulation_app.close()


if __name__ == "__main__":
    main()
