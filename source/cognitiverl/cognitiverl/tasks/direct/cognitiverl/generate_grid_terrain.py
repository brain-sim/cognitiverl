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
parser.add_argument(
    "--spawn_balls",
    action="store_true",
    help="Spawn test balls on top of terrain to test physics interaction",
)
parser.add_argument(
    "--num_balls",
    type=int,
    default=20,
    help="Number of balls to spawn when --spawn_balls is used",
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


def load_from_usd(usd_path: str):
    """Load from USD file."""
    print(f"📁 Loading terrain from: {usd_path}")

    try:
        # Method 1: Try direct USD spawning first
        cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg.func("/World/ground", cfg)
        print("✅ Terrain loaded successfully from USD!")
        return True

    except Exception as e1:
        print(f"⚠️ Manual USD loading failed: {e1}")
        return None


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
        # Generate the grid terrain with better parameters to reduce gaps
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
                        platform_width=4.0,  # Increased from 3.0 to reduce gaps
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

        # Now ensure the terrain has proper collision
        enable_terrain_collision()

        return terrain_importer

    except Exception as e:
        print(f"❌ Grid terrain generation failed: {e}")
        return None


def enable_terrain_collision():
    """Ensure terrain has proper physics collision enabled."""
    print("🔧 Enabling terrain collision...")

    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()

        # Find terrain prims
        terrain_paths = ["/World/ground", "/World/terrain", "/World/ground_plane"]

        collision_applied = False
        for terrain_path in terrain_paths:
            terrain_prim = stage.GetPrimAtPath(terrain_path)
            if terrain_prim.IsValid():
                print(f"🔧 Found terrain at: {terrain_path}")

                # Check if it's already a collision mesh or has children with meshes
                def apply_collision_to_mesh_prims(prim, path=""):
                    current_path = path if path else prim.GetPath()

                    # If this prim is a mesh, apply collision to it
                    if prim.GetTypeName() == "Mesh":
                        print(f"   🎯 Applying collision to mesh: {current_path}")

                        # Apply collision API
                        collision_api = UsdPhysics.CollisionAPI.Apply(prim)

                        # Apply mesh collision API for trimesh collision
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

                        # Force exact trimesh collision - no approximation
                        mesh_collision_api.CreateApproximationAttr().Set("none")
                        print(
                            f"     ✅ Using EXACT trimesh collision for {current_path}"
                        )

                        # Also check and print mesh statistics
                        try:
                            mesh_geom = prim
                            if hasattr(mesh_geom, "GetPointsAttr"):
                                points = mesh_geom.GetPointsAttr().Get()
                                if points:
                                    print(f"     📊 Mesh has {len(points)} vertices")
                        except Exception as stat_e:
                            print(f"     ⚠️ Could not get mesh stats: {stat_e}")

                        return True

                    # Recursively check children
                    collision_found = False
                    for child in prim.GetChildren():
                        if apply_collision_to_mesh_prims(child):
                            collision_found = True

                    return collision_found

                # Apply collision to all mesh children
                if apply_collision_to_mesh_prims(terrain_prim):
                    print(f"✅ Collision enabled for terrain at: {terrain_path}")
                    collision_applied = True
                else:
                    print(f"⚠️ No mesh geometry found in: {terrain_path}")

                if collision_applied:
                    break  # Only process the first valid terrain

        if not collision_applied:
            print("❌ No collision could be applied to any terrain!")

    except Exception as e:
        print(f"❌ Failed to enable terrain collision: {e}")
        import traceback

        traceback.print_exc()


def debug_terrain_collision():
    """Debug function to check terrain collision setup."""
    print("🔍 Debugging terrain collision setup...")

    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()

        def check_prim_collision(prim, depth=0):
            indent = "  " * depth
            path = prim.GetPath()
            type_name = prim.GetTypeName()

            # Check if prim has collision APIs applied
            has_collision = UsdPhysics.CollisionAPI.Get(stage, path)
            has_mesh_collision = UsdPhysics.MeshCollisionAPI.Get(stage, path)
            has_rigid_body = UsdPhysics.RigidBodyAPI.Get(stage, path)

            collision_info = []
            if has_collision:
                collision_info.append("Collision")
            if has_mesh_collision:
                collision_info.append("MeshCollision")
            if has_rigid_body:
                collision_info.append("RigidBody")

            collision_str = (
                f" [{', '.join(collision_info)}]" if collision_info else " [No Physics]"
            )

            print(f"{indent}- {path} ({type_name}){collision_str}")

            # Recursively check children
            for child in prim.GetChildren():
                check_prim_collision(child, depth + 1)

        world_prim = stage.GetPrimAtPath("/World")
        if world_prim.IsValid():
            check_prim_collision(world_prim)

    except Exception as e:
        print(f"❌ Debug failed: {e}")


def spawn_balls_on_terrain(terrain_size=None, num_balls=20):
    """Spawn physics balls on top of terrain to test terrain accuracy."""
    print(f"🏀 Spawning {num_balls} test balls on terrain...")

    try:
        import random

        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        # Get the current stage
        stage = omni.usd.get_context().get_stage()

        # Determine terrain bounds
        if terrain_size is None:
            terrain_size = args_cli.terrain_size

        # Calculate spawn area (slightly smaller than terrain to keep balls on terrain)
        x_min, x_max = -terrain_size[0] / 2 * 0.8, terrain_size[0] / 2 * 0.8
        y_min, y_max = -terrain_size[1] / 2 * 0.8, terrain_size[1] / 2 * 0.8
        spawn_height = 5.0  # Height above terrain to spawn balls

        # Ball properties
        ball_radius = 0.1
        ball_colors = [
            (1.0, 0.2, 0.2),  # Red
            (0.2, 1.0, 0.2),  # Green
            (0.2, 0.2, 1.0),  # Blue
            (1.0, 1.0, 0.2),  # Yellow
            (1.0, 0.2, 1.0),  # Magenta
            (0.2, 1.0, 1.0),  # Cyan
        ]

        # Create balls parent prim
        balls_prim = prim_utils.create_prim("/World/TestBalls", "Xform")

        for i in range(num_balls):
            # Random position above terrain
            x_pos = random.uniform(x_min, x_max)
            y_pos = random.uniform(y_min, y_max)
            z_pos = spawn_height + random.uniform(0, 2.0)  # Stagger heights

            ball_path = f"/World/TestBalls/Ball_{i:03d}"

            # Create sphere geometry
            ball_prim = prim_utils.create_prim(
                ball_path,
                "Sphere",
                translation=(x_pos, y_pos, z_pos),
                attributes={"radius": ball_radius},
            )

            # Add physics - make it a rigid body
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(ball_prim)

            # Set mass for realistic physics
            mass_api = UsdPhysics.MassAPI.Apply(ball_prim)
            mass_api.CreateMassAttr().Set(0.1)  # 100g ball

            # Add collision API
            collision_api = UsdPhysics.CollisionAPI.Apply(ball_prim)

            # For sphere collision, we need to use the sphere's geometry itself
            # The collision system will automatically use the sphere geometry for collision
            # We just need to ensure the sphere has the right radius
            sphere_geom = UsdGeom.Sphere(ball_prim)
            sphere_geom.CreateRadiusAttr().Set(ball_radius)

            # Add visual material (color) using the correct USD approach
            color = ball_colors[i % len(ball_colors)]

            # Create visual material using UsdShade properly
            visual_material_path = f"/World/TestBalls/BallVisualMaterial_{i}"
            visual_material_prim = prim_utils.create_prim(
                visual_material_path, "Material"
            )

            # Create the material and shader using UsdShade APIs
            material_api = UsdShade.Material(visual_material_prim)

            # Create shader
            shader_path = f"{visual_material_path}/Shader"
            shader_prim = prim_utils.create_prim(shader_path, "Shader")
            shader_api = UsdShade.Shader(shader_prim)

            # Set shader attributes using proper USD types and methods
            shader_api.CreateIdAttr().Set("UsdPreviewSurface")

            # Create inputs with correct types
            diffuse_input = shader_api.CreateInput(
                "diffuseColor", Sdf.ValueTypeNames.Color3f
            )
            diffuse_input.Set(Gf.Vec3f(*color))

            metallic_input = shader_api.CreateInput(
                "metallic", Sdf.ValueTypeNames.Float
            )
            metallic_input.Set(0.0)

            roughness_input = shader_api.CreateInput(
                "roughness", Sdf.ValueTypeNames.Float
            )
            roughness_input.Set(0.3)

            # Connect material to shader surface output
            material_api.CreateSurfaceOutput().ConnectToSource(
                shader_api.ConnectableAPI(), "surface"
            )

            # Bind visual material to ball
            material_binding_api = UsdShade.MaterialBindingAPI.Apply(ball_prim)
            material_binding_api.Bind(material_api)

            print(
                f"   🏀 Ball {i + 1}/{num_balls} spawned at ({x_pos:.2f}, {y_pos:.2f}, {z_pos:.2f})"
            )

        print("✅ All test balls spawned successfully!")
        print(
            "🎯 Watch the balls fall and interact with the terrain to test physics accuracy"
        )
        return True

    except Exception as e:
        print(f"❌ Failed to spawn balls: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main function to view terrain."""

    # Initialize simulation context with smaller physics time step
    # Disable GPU physics when spawning balls to avoid PhysX restrictions
    use_cpu_physics = args_cli.spawn_balls

    sim_params = {
        "use_gpu": not use_cpu_physics,  # Use CPU physics when spawning balls
        "use_gpu_pipeline": False,
        "use_flatcache": False,
        "use_fabric": False,
        "enable_scene_query_support": True,
    }

    if use_cpu_physics:
        print("🔧 Using CPU physics for ball spawning compatibility...")

    sim = SimulationContext(
        physics_dt=1.0
        / 120.0,  # Reduced from 1/60 to 1/120 for better collision detection
        rendering_dt=1.0 / 60.0,
        sim_params=sim_params,
        backend="torch",
        device="cuda:0" if not use_cpu_physics else "cpu",
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
            result = load_from_usd(terrain_path)
            terrain_loaded = result is not None
            # Handle both TerrainImporter object and boolean returns
            if isinstance(result, TerrainImporter):
                terrain_importer = result

            # Enable collision for loaded terrain too
            if terrain_loaded:
                enable_terrain_collision()
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

    # Debug terrain collision setup
    if args_cli.spawn_balls:
        debug_terrain_collision()

    # Spawn test balls if requested (after simulation is initialized)
    if args_cli.spawn_balls:
        # Wait longer for collision system to fully process the terrain
        import time

        print("⏳ Waiting for collision system to process terrain...")
        time.sleep(2.0)  # Increased wait time for proper collision setup

        # Force several physics steps to ensure collision is ready
        print("🔄 Running physics steps to settle collision system...")
        for _ in range(5):
            sim.step()

        spawn_balls_on_terrain(
            args_cli.terrain_size if args_cli.generate else None, args_cli.num_balls
        )

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
    if args_cli.spawn_balls:
        print("\n🏀 Ball Physics Test:")
        print("   • Watch how balls fall and settle on terrain")
        print("   • Balls should bounce slightly and roll naturally")
        print("   • Check if balls get stuck in terrain gaps")
        print("   • Observe collision accuracy with terrain surface")
        print("   • Using CPU physics with higher time resolution (120Hz)")
        print("   • NO backup plane - testing actual terrain collision accuracy")

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
