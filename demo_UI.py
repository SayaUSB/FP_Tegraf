import numpy as np
from openvino.runtime import Core
from initialize import initialize
import pygame
import sys
import threading
import time
import queue
import math

class FootstepPlanner:
    def __init__(self):
        # Initialize OpenVINO model
        self.core = Core()
        self.model = self.core.read_model("OpenVino_Model/any_obstacle_v1.xml")
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

        self.running = False
        self.planning_thread = None
        self.state_queue = queue.Queue()
        self.env = None

    def envInitialize(self):
        """Initialize Environment"""
        if self.env is None:
            self.env = initialize()
        self.obs, _ = self.env.reset(options=self.options)

    def run_plan(self, checkpoints):
        """Run the planning algorithm in a separate thread"""
        try:
            # Initialize environment
            self.envInitialize()
            
            # Add obstacles and checkpoints
            if hasattr(self.env, 'simulator'):
                self.env.simulator.checkpoints = checkpoints
                
            # Start from home position for the first checkpoint
            current_start = self.options["home"]
            
            # Run through all checkpoints
            for i in range(len(checkpoints)):
                # Update start position to the previous checkpoint
                if i > 0:
                    current_start = checkpoints[i-1]
                    
                # Set start and target positions
                self.options["start_foot_pose"] = current_start
                self.options["target_foot_pose"] = checkpoints[i]
                
                # Reset environment with new start and target
                self.envInitialize()
                
                # Run to current target
                self.run_to_target()
            
            # Return to home position from the last checkpoint
            self.options["start_foot_pose"] = checkpoints[-1]
            self.options["target_foot_pose"] = self.options["home"]
            self.envInitialize()
            self.run_to_target()
        finally:
            self.running = False

    def run_to_target(self):
        """Run the planner to the current target"""
        obs = self.obs
        terminated = False
        
        while self.running and not terminated:
            obs_input = np.expand_dims(obs, axis=0).astype(np.float32)
            result = self.compiled_model([obs_input])[self.output_layer]
            action = np.squeeze(result, axis=0)
            obs, _, terminated, _, info = self.env.step(action)
            
            # Send state to main thread for rendering
            self.state_queue.put({
                "foot_coord": info["Foot Coord"],
                "support_foot": info["Support Foot"],
                "terminated": terminated
            })
            
            time.sleep(0.01)  # Prevent flooding the queue

    def start_planning(self, checkpoints):
        """Start the planning process"""
        if self.running:
            return
            
        self.running = True
        # Clear any previous state
        while not self.state_queue.empty():
            self.state_queue.get()
            
        # Convert checkpoints to include orientation
        full_checkpoints = []
        for cp in checkpoints:
            if len(cp) == 2:
                full_checkpoints.append((cp[0], cp[1], 0))
            else:
                full_checkpoints.append(cp)
        
        self.planning_thread = threading.Thread(
            target=self.run_plan, 
            args=(full_checkpoints,),
            daemon=True
        )
        self.planning_thread.start()

def angular_distance(a, b):
    """Compute minimal angular difference in radians"""
    return min(abs(a - b), 2 * math.pi - abs(a - b))

def heuristic(a, b, angle_weight=1.0):
    """Heuristic that combines Euclidean distance and orientation difference"""
    dist = math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    theta_a = a[2] if len(a) > 2 else 0
    theta_b = b[2] if len(b) > 2 else 0
    angle_diff = angular_distance(theta_a, theta_b)
    
    return dist + angle_weight * angle_diff

def sort_checkpoints(checkpoints: list, start_pos: tuple, angle_weight=1.0):
    """Sort the checkpoints and adjust their orientation for minimal turns"""
    if start_pos is None:
        return checkpoints

    current = start_pos
    remaining = checkpoints.copy()
    ordered = []

    while remaining:
        nearest = min(remaining, key=lambda cp: heuristic(current, cp, angle_weight))
        remaining.remove(nearest)
        dx = nearest[0] - current[0]
        dy = nearest[1] - current[1]
        new_theta = math.atan2(dy, dx)
        ordered.append((nearest[0], nearest[1], new_theta))

        current = (nearest[0], nearest[1], new_theta)

    return ordered

if __name__ == "__main__":
    pygame.init()
    planner = FootstepPlanner()
    planner.envInitialize()
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                planner.running = False
                if planner.planning_thread and planner.planning_thread.is_alive():
                    planner.planning_thread.join(timeout=1.0)
                pygame.quit()
                sys.exit()
                
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if hasattr(planner.env, 'simulator'):
                    planner.env.simulator.handle_click(event.pos, event.button)
                    if planner.env.simulator.active_mode == "Set Start":
                        planner.env.options["home"] = np.array([event.pos[0], event.pos[1], 0], dtype=np.float32)
                    if planner.env.simulator.active_mode == "Plan Path":
                        sorted_cps = sort_checkpoints(
                            planner.env.simulator.checkpoints,
                            planner.env.simulator.start_pos
                        )
                        
                        planner.start_planning(sorted_cps)
        
        # Process state updates from planning thread
        while not planner.state_queue.empty():
            state = planner.state_queue.get()
            print(f"Foot Coord: {state['foot_coord']}, Support Foot: {state['support_foot']}")
            
            if state['terminated']:
                print("Target reached!")

        if hasattr(planner.env, 'simulator'):
            planner.env.simulator.render()
        
        clock.tick(120)