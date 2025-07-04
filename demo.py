import numpy as np
from openvino.runtime import Core
from initialize import initialize
import pygame

# Intialize OpenVINO model
class FootstepPlanner:
    def __init__(self):
        # Initialize OpenVINO model
        self.core = Core()
        self.model = self.core.read_model("OpenVino_Model/any_obstacle_v1.xml") # Give the right directory for the model
        self.compiled_model = self.core.compile_model(self.model, "CPU")
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)

        self.options = {
            # Maximum steps
            "max_dx_forward": 0.08,  # [m]
            "max_dx_backward": 0.03,  # [m]
            "max_dy": 0.04,  # [m]
            "max_dtheta": np.deg2rad(20),  # [rad]
            # Target tolerance
            "tolerance_distance": 1,  # [m]
            "tolerance_angle": float("inf"),  # [rad]
            # Do we include collisions with the ball?
            "has_obstacle": False,
            "obstacle_max_radius": 0.25,  # [m]
            "obstacle_radius": None,  # [m]
            "obstacle_position": np.array([0, 0], dtype=np.float32),  # [m,m]
            # Which foot is targeted (any, left or right)
            "foot": "any",
            # Foot geometry
            "foot_length": 0.7,  # [m]
            "foot_width": 0.04,  # [m]
            "feet_spacing": 0.15,  # [m]
            # Add reward shaping term
            "shaped": True,
            # If True, the goal will be sampled in a 4x4m area, else it will be fixed at (0,0)
            "multi_goal": False,
            "start_foot_pose": np.array([0,0,0], dtype=np.float32),
            "target_foot_pose": np.array([8,6,0], dtype=np.float32),
            "panjang": 8, # [m]
            "lebar": 6, # [m]
            "home": np.array([0,0,0], dtype=np.float32)
        }

    def envInitialize(self):
        """Initialize Environment"""
        self.env = initialize()
        # self.env = TimeLimit(self.env, max_episode_steps=1000)
        self.obs, _ = self.env.reset(options=self.options)

    def main(self):
        obs = self.obs
        while True:
            obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
            result = self.compiled_model([obs_input])[self.output_layer]
            action = np.squeeze(result, axis=0)
            obs, _, terminated, _, info = self.env.step(action)

            if terminated:
                self.env.close()
                break
        
            print(info["Foot Coord"], info["Support Foot"])
            self.env.simulator.path = self.env.simulator.run_to_checkpoint_and_back()
            self.env.render()

def sort_checkpoints(checkpoints: list, start_pos: tuple):
    """Sort the checkpoints that need to be visited"""
    if start_pos is None:
        return
    
    current = start_pos
    ordered = []

    while checkpoints:
        nearest = min(checkpoints, key=lambda cp: heuristic(current, cp))
        ordered.append(nearest)
        checkpoints.remove(nearest)
        current = nearest

    return ordered

def heuristic(a, b):
    """Heuristic Euclidean Distance"""
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5

if __name__ == "__main__":
    ashioto = FootstepPlanner()
    ashioto.options["start_foot_pose"] = list(map(float, input("Start Position (X, Y, Z): ").split()))
    ashioto.options["home"] = ashioto.options["start_foot_pose"]
    checkpoints = [(3,3,0),(1,1,-3), (-1,-3,-2)]
    obstacles = [(2,2)]
    checkpoints = sort_checkpoints(checkpoints, ashioto.options["home"][:2])
    for i in range(len(checkpoints)):
        if i>0:
            ashioto.options["start_foot_pose"] = checkpoints[i-1]
        ashioto.options["target_foot_pose"] = checkpoints[i]
        ashioto.envInitialize()
        ashioto.env.add_checkpoints(checkpoints)
        ashioto.env.add_obstacles(obstacles)
        ashioto.main()
    
    # Kembali posisi awal
    ashioto.options["start_foot_pose"] = checkpoints[-1]
    ashioto.options["target_foot_pose"] = ashioto.options["home"]
    ashioto.envInitialize()
    ashioto.env.add_checkpoints(checkpoints)
    ashioto.env.add_obstacles(obstacles)
    ashioto.main()
