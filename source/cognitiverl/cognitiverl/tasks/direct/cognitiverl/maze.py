import numpy as np
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import FixedCuboid


WALL_CFG = {
    "wall_thickness": 2.0,
    "wall_height": 3.0,
    "wall_color": (0.2, 0.3, 0.8),
    "physics_material": {
        "dynamic_friction": 1.0,
        "static_friction": 1.5,
        "restitution": 0.1
    }
}

def create_walls(env_name, env_origin, room_size):
    """Create walls for an environment"""
    wall_thickness = WALL_CFG["wall_thickness"]
    wall_height = WALL_CFG["wall_height"]
    wall_position = room_size / 2
    
    # Create physics material for walls
    PhysicsMaterial(
        prim_path="/World/physics_material/wall_material",
        dynamic_friction=WALL_CFG["physics_material"]["dynamic_friction"],
        static_friction=WALL_CFG["physics_material"]["static_friction"],
        restitution=WALL_CFG["physics_material"]["restitution"],
    )
    
    # Convert CUDA tensor to CPU before using in NumPy
    origin_cpu = env_origin.cpu().numpy()
    
    # Create the four walls
    walls = {}
    
    # North wall (top)
    walls["north_wall"] = FixedCuboid(
        prim_path=f"/World/envs/{env_name}/walls/north_wall",
        position=np.array([origin_cpu[0], origin_cpu[1] + wall_position, wall_height / 2]),
        scale=np.array([room_size + wall_thickness, wall_thickness, wall_height]),
        color=np.array(WALL_CFG["wall_color"]),
    )
    
    # South wall (bottom)
    walls["south_wall"] = FixedCuboid(
        prim_path=f"/World/envs/{env_name}/walls/south_wall",
        position=np.array([origin_cpu[0], origin_cpu[1] - wall_position, wall_height / 2]),
        scale=np.array([room_size + wall_thickness, wall_thickness, wall_height]),
        color=np.array(WALL_CFG["wall_color"]),
    )
    
    # East wall (right)
    walls["east_wall"] = FixedCuboid(
        prim_path=f"/World/envs/{env_name}/walls/east_wall",
        position=np.array([origin_cpu[0] + wall_position, origin_cpu[1], wall_height / 2]),
        scale=np.array([wall_thickness, room_size + wall_thickness, wall_height]),
        color=np.array(WALL_CFG["wall_color"]),
    )
    
    # West wall (left)
    walls["west_wall"] = FixedCuboid(
        prim_path=f"/World/envs/{env_name}/walls/west_wall",
        position=np.array([origin_cpu[0] - wall_position, origin_cpu[1], wall_height / 2]),
        scale=np.array([wall_thickness, room_size + wall_thickness, wall_height]),
        color=np.array(WALL_CFG["wall_color"]),
    )
    
    return walls