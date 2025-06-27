import isaaclab.sim as sim_utils


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


def enable_terrain_collision():
    """Ensure terrain has proper physics collision enabled."""
    print("🔧 Enabling terrain collision...")

    try:
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()

        # Find terrain prims
        terrain_paths = ["/World/ground"]

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
