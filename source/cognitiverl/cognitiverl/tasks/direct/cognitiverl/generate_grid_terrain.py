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
    default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "custom_assets/grid_terrain.usd"
    ),
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
    default=os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "custom_assets/grid_terrain.usd"
    ),
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
    default=10,
    help="Number of balls to spawn when --spawn_balls is used",
)
parser.add_argument(
    "--collision_method",
    type=str,
    choices=["auto", "heightfield", "convex", "trimesh"],
    default="auto",
    help="Collision method: auto (chooses based on terrain size), heightfield (most memory efficient), "
    "convex (balanced), trimesh (most accurate but memory intensive)",
)
parser.add_argument(
    "--force_memory_efficient",
    action="store_true",
    help="Force memory-efficient settings for very large terrains",
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
    """Load from USD file with physics properties intact."""
    print(f"📁 Loading terrain from: {usd_path}")

    try:
        # Load USD file and verify physics properties are intact
        cfg = sim_utils.UsdFileCfg(usd_path=usd_path)
        cfg.func("/World/ground", cfg)

        # Verify physics properties are loaded correctly
        verify_loaded_physics_properties()

        print("✅ Terrain loaded successfully from USD with physics properties!")
        return True

    except Exception as e1:
        print(f"⚠️ USD loading failed: {e1}")
        import traceback

        traceback.print_exc()
        return None


def verify_loaded_physics_properties():
    """Verify that loaded terrain has proper physics properties."""
    print("🔍 Verifying loaded physics properties...")

    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        terrain_paths = ["/World/ground", "/World/terrain", "/World/ground_plane"]

        physics_found = False
        for terrain_path in terrain_paths:
            terrain_prim = stage.GetPrimAtPath(terrain_path)
            if terrain_prim.IsValid():
                print(f"🔍 Checking physics for: {terrain_path}")

                def check_physics_recursive(prim, depth=0):
                    indent = "  " * depth
                    path = prim.GetPath()

                    # Check physics APIs
                    has_collision = UsdPhysics.CollisionAPI.Get(stage, path)
                    has_mesh_collision = UsdPhysics.MeshCollisionAPI.Get(stage, path)
                    has_rigid_body = UsdPhysics.RigidBodyAPI.Get(stage, path)

                    if has_collision or has_mesh_collision or has_rigid_body:
                        apis = []
                        if has_collision:
                            apis.append("Collision")
                        if has_mesh_collision:
                            apis.append("MeshCollision")
                            # Check approximation method
                            approx_attr = has_mesh_collision.GetApproximationAttr()
                            if approx_attr:
                                approx_value = approx_attr.Get()
                                if approx_value:
                                    apis.append(f"Approx:{approx_value}")
                        if has_rigid_body:
                            apis.append("RigidBody")

                        print(f"{indent}✅ {path}: [{', '.join(apis)}]")
                        return True

                    # Check children
                    found_in_children = False
                    for child in prim.GetChildren():
                        if check_physics_recursive(child, depth + 1):
                            found_in_children = True

                    return found_in_children

                if check_physics_recursive(terrain_prim):
                    physics_found = True
                    print(f"✅ Physics properties verified for: {terrain_path}")
                    break

        if physics_found:
            print("✅ All physics properties loaded correctly - no need to re-enable!")
        else:
            print("⚠️ No physics properties found in loaded terrain")

        return physics_found

    except Exception as e:
        print(f"❌ Failed to verify physics properties: {e}")
        return False


def save_terrain_as_usd(save_path: str):
    """Save the current terrain as USD file with all physics properties preserved."""
    print(f"💾 Saving terrain to: {save_path}")

    try:
        import omni.usd
        from pxr import Usd, UsdPhysics

        # Get the current stage
        stage = omni.usd.get_context().get_stage()

        # Create a new layer for export
        export_layer = Usd.Stage.CreateNew(save_path)

        # Copy terrain prims to export stage, preserving all APIs and properties
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

                # CRITICAL: Copy physics APIs
                def copy_physics_apis(source_prim, dest_prim):
                    """Copy all physics-related APIs to preserve collision properties."""
                    try:
                        # Check and copy Collision API
                        if UsdPhysics.CollisionAPI.Get(stage, source_prim.GetPath()):
                            collision_api = UsdPhysics.CollisionAPI.Apply(dest_prim)
                            print("     ✅ Copied CollisionAPI")

                        # Check and copy MeshCollision API
                        mesh_collision = UsdPhysics.MeshCollisionAPI.Get(
                            stage, source_prim.GetPath()
                        )
                        if mesh_collision:
                            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(
                                dest_prim
                            )
                            # Copy approximation setting
                            approx_attr = mesh_collision.GetApproximationAttr()
                            if approx_attr and approx_attr.Get():
                                mesh_collision_api.CreateApproximationAttr().Set(
                                    approx_attr.Get()
                                )
                                print(
                                    f"     ✅ Copied MeshCollisionAPI with approximation: {approx_attr.Get()}"
                                )

                        # Check and copy RigidBody API
                        if UsdPhysics.RigidBodyAPI.Get(stage, source_prim.GetPath()):
                            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(dest_prim)
                            print("     ✅ Copied RigidBodyAPI")

                        # Check and copy Mass API
                        mass_api = UsdPhysics.MassAPI.Get(stage, source_prim.GetPath())
                        if mass_api:
                            dest_mass_api = UsdPhysics.MassAPI.Apply(dest_prim)
                            # Copy mass value if it exists
                            mass_attr = mass_api.GetMassAttr()
                            if mass_attr and mass_attr.Get() is not None:
                                dest_mass_api.CreateMassAttr().Set(mass_attr.Get())
                                print("     ✅ Copied MassAPI")

                    except Exception as api_e:
                        print(f"⚠️ Error copying physics APIs: {api_e}")

                # Copy physics APIs for the main prim
                copy_physics_apis(terrain_prim, export_prim)

                # Copy children recursively WITH physics APIs
                def copy_children_with_physics(source_prim, dest_stage, dest_path):
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

                            # CRITICAL: Copy physics APIs for children too
                            copy_physics_apis(child, dest_child)

                            # Recursively copy grandchildren
                            copy_children_with_physics(child, dest_stage, child_path)
                        except Exception as child_e:
                            print(f"⚠️ Could not copy child {child_name}: {child_e}")

                copy_children_with_physics(terrain_prim, export_layer, root_path)

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
            print("✅ All physics properties preserved in USD file!")
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
    """Generate terrain on the fly with built-in physics properties for memory efficiency."""
    print(f"🏗️ Generating terrain of size {args_cli.terrain_size}...")

    try:
        # For large terrains, use height field terrain instead of mesh terrain
        # This is much more memory efficient
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
            # Enable physics properties directly in the terrain configuration
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.06, 0.08, 0.1),  # Dark blue-gray color
                metallic=0.0,
                roughness=0.8,
            ),
            max_init_terrain_level=0,
            debug_vis=False,
        )

        terrain_importer = TerrainImporter(terrain_cfg)

        # CRITICAL: Wait for terrain mesh to be fully generated before applying collision
        print("⏳ Waiting for terrain mesh generation to complete...")
        import time

        time.sleep(1.0)  # Give terrain generation time to complete

        # Force a few simulation steps to ensure geometry is ready
        try:
            # Get simulation context if available
            from isaacsim.core.api.simulation_context import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context:
                print("🔄 Running simulation steps to finalize terrain geometry...")
                for _ in range(3):
                    sim_context.step()
        except:
            print("⚠️ Could not access simulation context for geometry finalization")

        # Now apply collision after geometry is ready
        setup_collision_by_method(args_cli.collision_method)

        print("✅ Grid terrain generated successfully with built-in physics!")
        return terrain_importer

    except Exception as e:
        print(f"❌ Grid terrain generation failed: {e}")
        import traceback

        traceback.print_exc()
        return None


def generate_large_terrain_efficient(terrain_size):
    """Generate very large terrain (e.g., 1000x1000) using heightfield collision for memory efficiency."""
    print(
        f"🏗️ Generating LARGE terrain ({terrain_size[0]}x{terrain_size[1]}) with heightfield collision..."
    )

    # For terrains larger than 100x100, use heightfield collision
    if terrain_size[0] > 10 or terrain_size[1] > 10:
        print(
            "📊 Large terrain detected - using heightfield collision for memory efficiency"
        )

        try:
            # Use heightfield terrain generation - much more memory efficient
            terrain_cfg = TerrainImporterCfg(
                prim_path="/World/ground",
                terrain_type="generator",
                terrain_generator=terrain_gen.TerrainGeneratorCfg(
                    seed=1,
                    use_cache=True,
                    size=tuple(terrain_size),
                    num_rows=1,
                    num_cols=1,
                    sub_terrains={
                        # Use heightfield terrain instead of mesh - MUCH more memory efficient
                        "flat_terrain": terrain_gen.HfDiscreteObstaclesTerrainCfg(
                            proportion=1.0,
                            obstacle_height_range=(0.01, 0.07),
                            obstacle_width_range=(0.3, 0.6),
                            num_obstacles=10,
                        ),
                    },
                ),
                # Enable built-in physics
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.06, 0.08, 0.1),
                    metallic=0.0,
                    roughness=0.8,
                ),
                max_init_terrain_level=0,
                debug_vis=False,
            )

            terrain_importer = TerrainImporter(terrain_cfg)

            # For very large terrains, use heightfield collision instead of mesh collision
            setup_collision_by_method(args_cli.collision_method, terrain_size)

            print("✅ Large terrain generated with efficient collision!")
            return terrain_importer

        except Exception as e:
            print(f"❌ Large terrain generation failed: {e}")
            import traceback

            traceback.print_exc()
            return None
    else:
        # For smaller terrains, use the regular method
        return generate_terrain()


def setup_heightfield_collision_for_large_terrain():
    """Setup heightfield collision for very large terrains - extremely memory efficient."""
    print("🔧 Setting up heightfield collision for large terrain...")

    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        terrain_paths = ["/World/ground", "/World/terrain", "/World/ground_plane"]

        for terrain_path in terrain_paths:
            terrain_prim = stage.GetPrimAtPath(terrain_path)
            if terrain_prim.IsValid():
                print(f"🔧 Found terrain at: {terrain_path}")

                def setup_heightfield_collision(prim):
                    if prim.GetTypeName() == "Mesh":
                        print(
                            f"   🎯 Setting up heightfield collision for: {prim.GetPath()}"
                        )

                        # Apply collision API
                        collision_api = UsdPhysics.CollisionAPI.Apply(prim)

                        # Use heightfield collision - most memory efficient for large terrains
                        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

                        # Heightfield collision is extremely memory efficient for large flat terrains
                        # It uses a 2D height array instead of full 3D mesh
                        mesh_collision_api.CreateApproximationAttr().Set("heightfield")

                        print(
                            "     ✅ Using heightfield collision (maximum memory efficiency)"
                        )
                        return True

                    # Check children
                    for child in prim.GetChildren():
                        if setup_heightfield_collision(child):
                            return True
                    return False

                if setup_heightfield_collision(terrain_prim):
                    print(f"✅ Heightfield collision setup for: {terrain_path}")
                    break

    except Exception as e:
        print(f"❌ Failed to setup heightfield collision: {e}")
        print("⚠️ Falling back to convex hull collision...")
        # Fallback to convex hull if heightfield fails
        setup_collision_by_method()


def setup_collision_by_method(method="auto", terrain_size=None):
    """Setup collision using the specified method."""
    if terrain_size is None:
        terrain_size = args_cli.terrain_size

    # For mesh grid terrain, we need trimesh collision for accuracy with individual cells
    if method == "auto":
        # Mesh grid terrain needs trimesh collision for individual cells to work properly
        method = "trimesh"
        print(
            "🎯 Auto-selected TRIMESH collision for mesh grid terrain (required for individual cells)"
        )

    # For mesh grid terrain, force trimesh collision - convex doesn't work well with small cells
    if method == "convex" or method == "heightfield":
        method = "trimesh"
        print(
            "🎯 Switching to TRIMESH collision - required for mesh grid terrain individual cells"
        )

    print(f"🔧 Setting up {method} collision for mesh grid terrain...")

    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()
        terrain_paths = ["/World/ground", "/World/terrain", "/World/ground_plane"]

        collision_applied = False
        total_meshes_found = 0
        total_collisions_applied = 0

        for terrain_path in terrain_paths:
            terrain_prim = stage.GetPrimAtPath(terrain_path)
            if terrain_prim.IsValid():
                print(f"🔧 Found terrain at: {terrain_path}")

                def apply_collision_to_all_meshes(prim, depth=0):
                    """Recursively apply collision to all mesh primitives."""
                    nonlocal total_meshes_found, total_collisions_applied
                    indent = "  " * depth
                    current_path = prim.GetPath()
                    prim_type = prim.GetTypeName()

                    collision_applied_here = False

                    if prim_type == "Mesh":
                        total_meshes_found += 1
                        print(
                            f"{indent}🎯 Applying {method} collision to mesh #{total_meshes_found}: {current_path}"
                        )

                        try:
                            # Apply collision API
                            collision_api = UsdPhysics.CollisionAPI.Apply(prim)
                            mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(prim)

                            # Force trimesh collision for individual mesh cells
                            mesh_collision_api.CreateApproximationAttr().Set("none")
                            print(f"{indent}     ✅ Applied EXACT trimesh collision")

                            # Verify collision was applied
                            if UsdPhysics.CollisionAPI.Get(stage, current_path):
                                total_collisions_applied += 1
                                collision_applied_here = True
                                print(
                                    f"{indent}     ✅ Collision verified for mesh #{total_meshes_found}"
                                )
                            else:
                                print(
                                    f"{indent}     ❌ Collision verification failed for mesh #{total_mes