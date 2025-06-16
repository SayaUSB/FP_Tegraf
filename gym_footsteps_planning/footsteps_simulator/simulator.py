from matplotlib.colors import to_rgb
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import pygame
import heapq

if __name__ == "__main__":
    import transform as tr
else:
    import gym_footsteps_planning.footsteps_simulator.transform as tr

def other_foot(foot: str) -> str:
    """
    Returns the other foot than a given one
    """
    return "left" if foot == "right" else "right"


class Simulator:
    def __init__(self):
        # The current foot supporting the robot
        self.support_foot: str = "left"
        # Transformation matrix describing the pose of the support foot
        self.T_world_support: np.ndarray = np.eye(3)
        # Feet size
        self.foot_length: float = 0.14  # [m]
        self.foot_width: float = 0.08  # [m]
        # Distance between the feet
        self.feet_spacing: float = 0.15  # [m]

        # Initializing the robot
        self.init(0, 0, 0, 0, 0, 0)

        # Rendering parameters
        self.screen = None
        # self.size: tuple = (1024, 800)
        # self.pixels_per_meter: int = 200
        # Demo mode
        self.size = (1920, 1080)
        self.pixels_per_meter = 150

        self.left_color: tuple = (221, 103, 75)
        self.right_color: tuple = (75, 164, 221)

        # Obstacle position
        self.obstacles: list = []

        # Desired goal
        self.desired_goal = None

        # Optional path to draw
        self.path = None

        # Extra footsteps to draw
        self.extra_footsteps = []

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.size, 0, 32)

        self.checkpoints = [] # List of (x,y) tuples
        self.start_pos = None
        self.start_icon = pygame.image.load("Picture/pos.png").convert_alpha()
        self.start_icon = pygame.transform.scale(self.start_icon, (128, 128))
        self.checkpoint_icon = pygame.image.load("Picture/rumah.png").convert_alpha()
        self.checkpoint_icon = pygame.transform.scale(self.checkpoint_icon, (64, 64))
        self.obstacle_icon = pygame.image.load("Picture/batu.png").convert_alpha()
        self.obstacle_icon = pygame.transform.scale(self.obstacle_icon, (64, 64))

        # UI Button
        self.button_font = pygame.font.SysFont("Arial", 24)
        self.buttons = {
            "Add Obstacle": pygame.Rect(150, 20, 180, 40),
            "Add Checkpoint": pygame.Rect(150, 80, 180, 40),
            "Set Start": pygame.Rect(150, 140, 180, 40),
            "Plan Path": pygame.Rect(150, 200, 180, 40),
            "Reset": pygame.Rect(150, 260, 180, 40),
        }
        self.active_mode = "Add Obstacle"

        # Direction
        self.directions = [
            (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
            for angle in range(0, 360, 45)
        ]
        
        # FPS tracking
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 24)
        self.fps = 0
        self.last_time = time.time()

    def draw_buttons(self):
        for label, rect in self.buttons.items():
            color = (0, 200, 0) if self.active_mode == label else (180, 180, 180)
            pygame.draw.rect(self.screen, color, rect)
            text_surf = self.button_font.render(label, True, (0, 0, 0))
            self.screen.blit(text_surf, (rect.x + 10, rect.y + 5))

    def handle_click(self, pos, button):
        for label, rect in self.buttons.items():
            if rect.collidepoint(pos):
                self.active_mode = label
                if label == "Plan Path":
                    self.plan_path()
                    return True
                if label == "Reset":
                    self.reset_simulator()
                return

        world_pos = np.linalg.inv(self.T_screen_world) @ np.array([pos[0], pos[1], 1])
        x, y = round(world_pos[0], 2), round(world_pos[1], 2)

        if button == 1:  # Left click
            if self.active_mode == "Add Obstacle":
                self.add_obstacle((x, y), 0.5)
            elif self.active_mode == "Add Checkpoint":
                self.add_checkpoint(x, y)
            elif self.active_mode == "Set Start":
                self.init(x, y, 0, x, y, 0)
        elif button == 3:  #  Right click
            self.remove_nearest_object(x, y)

    def remove_nearest_object(self, x, y, threshold=0.5):
        target = (x, y)

        # Remove obstacle
        for obs in self.obstacles:
            if self.heuristic(obs[0], target) <= threshold:
                self.obstacles.remove(obs)
                return

        # Remove checkpoint
        for cp in self.checkpoints:
            if self.heuristic(cp, target) <= threshold:
                self.checkpoints.remove(cp)
                return

    def init(self, x: float, y: float, yaw: float, x_home: float, y_home: float, degree: float, start_support_foot: str = "left"):
        """
        Initializes the robot with the given foot at a given support position
        """
        self.support_foot = start_support_foot
        self.T_world_support = tr.frame(x, y, yaw)

        self.footsteps: list = []
        self.start_pos = (x, y)
        self.home = (x_home, y_home)
        self.save_footstep()

    def add_checkpoint(self, x: float, y: float):
        """Adds a checkpoint to the environment"""
        self.checkpoints.append([x, y])

    def clear_obstacles(self):
        """
        Clears the list of obstacles
        """
        self.obstacles = []

    def add_obstacle(self, position: np.ndarray, radius: float, color: tuple = (0, 0, 0, 0)):
        """
        Adds an obstacle to the environment
        """
        self.obstacles.append((position, radius, color))

    def T_world_neutral(self) -> np.ndarray:
        """
        Transformation from the neutral foot to the world (3x3 matrix)
        """
        offset_sign = 1 if self.support_foot == "right" else -1
        T_support_neutral = tr.frame(0, offset_sign * self.feet_spacing, 0)
        self.T_world_support

        return self.T_world_support @ T_support_neutral

    def P_world_neutral(self):
        return self.T_world_neutral()[:2, 2]

    def support_pose(self):
        """
        Alias for foot_pose("support")
        """
        return self.foot_pose("support")

    def neutral_pose(self):
        """
        Alias for foot_pose("neutral")
        """
        return self.foot_pose("neutral")

    def foot_pose(self, foot="support") -> list:
        """
        Returns the pose (x, y, yaw) of a given foot (can be "left", "right", "support", or "neutral")
        """
        if foot == "left":
            foot = "support" if self.support_foot == "left" else "neutral"
        if foot == "right":
            foot = "support" if self.support_foot == "right" else "neutral"

        if foot == "support":
            T_world_foot = self.T_world_support
        else:
            T_world_foot = self.T_world_neutral()

        position = T_world_foot[:2, 2]
        yaw = math.atan2(T_world_foot[1, 0], T_world_foot[0, 0])

        return [*position, yaw]

    def save_footstep(self):
        """
        Appends a footstep to the list of footsteps to draw
        """
        self.footsteps.append((self.support_foot, self.T_world_support.copy()))

    def step(self, dx: float, dy: float, dtheta: float):
        """
        Takes a step (updates the current support foot)
        """
        T_neutral_target = tr.frame(dx, dy, dtheta)
        self.T_world_support = self.T_world_neutral() @ T_neutral_target
        self.support_foot = other_foot(self.support_foot)

        self.save_footstep()

    def set_desired_goal(self, x: float, y: float, yaw: float, foot: str):
        self.desired_goal = (x, y, yaw, foot)

    def draw_path(self, surface):

        if self.path is None:
            return

        for i in range(len(self.path) - 1):
            progress = i / (len(self.path) - 1)
            color = tuple(int(255*x) for x in to_rgb(plt.cm.cool(progress)))
            
            ptA = self.T_screen_world @ np.array([self.path[i][0], self.path[i][1], 1]).T
            ptB = self.T_screen_world @ np.array([self.path[i + 1][0], self.path[i + 1][1], 1]).T
            pygame.draw.line(surface, color, (ptA[0], ptA[1]), (ptB[0], ptB[1]), 5)

    def draw_footstep(
        self,
        side: str,
        T_world_foot: np.ndarray,
        ratio: float,
        surface,
        fill: bool = True,
    ):
        """
        Draws a footstep
        """

        color = self.right_color if side == "right" else self.left_color

        points = [
            np.array([-self.foot_length / 2, self.foot_width / 2, 1]).T,
            np.array([self.foot_length / 2, self.foot_width / 2, 1]).T,
            np.array([self.foot_length * 0.4, 0, 1]).T,
            np.array([self.foot_length / 2, -self.foot_width / 2, 1]).T,
            np.array([-self.foot_length / 2, -self.foot_width / 2, 1]).T,
            np.array([-self.foot_length * 0.55, 0, 1]).T,
        ]
        result = []
        for point in points:
            pt = self.T_screen_world @ T_world_foot @ point
            result.append((float(pt[0]), float(pt[1])))

        alpha = ratio * 250
        color += (alpha,)

        if fill:
            try:
                pygame.draw.polygon(surface, color, result)
                pygame.draw.aalines(surface, (0, 0, 0, int(alpha)), True, result)
            except:
                pass
        else:
            try:
                pygame.draw.polygon(surface, color, result, width=3)
            except:
                pass

    def draw_grid(self, xmin: float = -4, xmax: float = 4, step: float = 0.25):
        """
        Draws the grid with the given step
        """

        for z in np.arange(xmin, xmax, step):
            if abs(z - int(z)) < 0.01:
                color = (100, 100, 100)
            else:
                color = (200, 200, 200)
            ptA = self.T_screen_world @ np.array([z, xmin, 1]).T
            ptB = self.T_screen_world @ np.array([z, xmax, 1]).T
            pygame.draw.line(self.screen, color, (ptA[0], ptA[1]), (ptB[0], ptB[1]), width=2)

            ptA = self.T_screen_world @ np.array([xmin, z, 1]).T
            ptB = self.T_screen_world @ np.array([xmax, z, 1]).T
            pygame.draw.line(self.screen, color, (ptA[0], ptA[1]), (ptB[0], ptB[1]), width=2)
            
    def draw_fps_counter(self):
        """Draws the FPS counter in the top-right corner"""
        fps_text = f"FPS: {self.fps:.1f}"
        text_surf = self.font.render(fps_text, True, (0, 0, 0))
        text_rect = text_surf.get_rect(topright=(self.size[0] - 20, 20))
        pygame.draw.rect(self.screen, (200, 200, 200, 180), 
                         (text_rect.left - 10, text_rect.top - 5, 
                          text_rect.width + 20, text_rect.height + 10))
        self.screen.blit(text_surf, text_rect)

    def render(self):
        """
        Renders the currently stored footsteps
        """
        # Calculate FPS
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        self.fps = 1.0 / dt if dt > 0 else 0

        self.T_screen_world = tr.translation(self.size[0] / 2, self.size[1] / 2)
        self.T_screen_world[0, 0] = self.pixels_per_meter
        self.T_screen_world[1, 1] = -self.pixels_per_meter

        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode(self.size, 0, 32)

        pygame.draw.rect(self.screen, (255, 255, 255), (0, 0, self.size[0], self.size[1]))

        self.draw_grid()

        for obstacle in self.obstacles:
            position, radius, color = obstacle

            P_obstacle_world = np.array([position[0], position[1], 1]).T
            P_ball_screen = self.T_screen_world @ P_obstacle_world
            tmp = pygame.Surface(self.size, pygame.SRCALPHA)
            tmp.set_colorkey((0, 0, 0))
            pygame.draw.circle(
                tmp,
                color,
                (int(P_ball_screen[0]), int(P_ball_screen[1])),
                int(radius * self.pixels_per_meter),
            )

            self.screen.blit(tmp, (0, 0))

        # Draw the goal support foot (if any)
        if self.desired_goal is not None:
            self.draw_footstep(
                self.desired_goal[3],
                tr.frame(*self.desired_goal[:3]),
                1,
                self.screen,
                fill=False,
            )

        # Draw path (if any)
        self.draw_path(self.screen)

        surface = pygame.Surface(self.size, pygame.SRCALPHA)
        surface.set_colorkey((0, 0, 0))
        index = 0
        for side, T_world_foot in self.footsteps:
            index += 1
            self.draw_footstep(side, T_world_foot, pow(index / len(self.footsteps), 3), surface)

        for extra_footstep in self.extra_footsteps:
            side, pose = extra_footstep
            self.draw_footstep(side, tr.frame(*pose), 1, surface, fill=False)

        if self.home is not None:
            pt_start = self.T_screen_world @ np.array([self.home[0], self.home[1], 1]).T
            rect = self.start_icon.get_rect(center=(int(pt_start[0]), int(pt_start[1])))
            self.screen.blit(self.start_icon, rect)

        for checkpoint in self.checkpoints:
            pt = self.T_screen_world @ np.array([checkpoint[0], checkpoint[1], 1]).T
            rect = self.checkpoint_icon.get_rect(center=(int(pt[0]), int(pt[1])))
            self.screen.blit(self.checkpoint_icon, rect)

        for obstacle in self.obstacles:
            pt = self.T_screen_world @ np.array([obstacle[0][0], obstacle[0][1], 1]).T
            rect = self.checkpoint_icon.get_rect(center=(int(pt[0]), int(pt[1])))
            self.screen.blit(self.obstacle_icon, rect)

        self.draw_buttons()
        self.screen.blit(surface, (0, 0))
        
        # Draw FPS counter
        self.draw_fps_counter()
        
        pygame.display.flip()

    def heuristic(self, a, b):
        """Heuristic Euclidean Distance"""
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**0.5
    
    def astar(self, start, goal, obstacles, resolution=0.2, angle_step=30):
        "Pathfind through multiple checkpoints with 360° directions"

        # Check if start and goal are too close
        if self.heuristic(start, goal) < resolution:
            return [goal]

        open_set = []
        heapq.heappush(open_set, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}

        visited = set()

        obstacle_grid = {}
        for obs in obstacles:
            key = (int(obs[0][0]/1), int(obs[0][1]/1))  # Grid 1x1 meter
            if key not in obstacle_grid:
                obstacle_grid[key] = []
            obstacle_grid[key].append(obs)

        while open_set:
            _, current = heapq.heappop(open_set)
            current = tuple(current)

            if current in visited:
                continue
            visited.add(current)

            if self.heuristic(current, goal) < resolution:
                path = [goal]
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.reverse()
                return path
            
            for dx, dy in self.directions:
                neighbor = (current[0] + dx * resolution, current[1] + dy * resolution)
                
                # Check local grid for obstacles
                neighbor_key = (int(neighbor[0]), int(neighbor[1]))
                collision = False
                if neighbor_key in obstacle_grid:
                    for obs in obstacle_grid[neighbor_key]:
                        if self.heuristic(neighbor, obs[0]) <= obs[1]:
                            collision = True
                            break
                if collision:
                    continue

            # for dx, dy in self.directions:
                neighbor = (
                    round(current[0] + dx * resolution, 2),
                    round(current[1] + dy * resolution, 2)
                )
                if any(self.heuristic(neighbor, obs[0]) <= obs[1] for obs in obstacles):
                    continue
                tentative_g = g_score[current] + self.heuristic(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f, neighbor))
                    came_from[neighbor] = current

        return []

    def run_to_checkpoint_and_back(self):
        start = self.home
        remaining = self.checkpoints.copy()
        all_path = []
        current = start

        while remaining:
            # Find the nearest checkpoint
            nearest = min(remaining, key=lambda cp: self.heuristic(current, cp))
            path = self.astar(current, nearest, self.obstacles)
            if path:
                all_path += path[1:]  # Avoid duplicate pos
                current = nearest
            remaining.remove(nearest)

        # Back to the starting pos
        # back_path = self.astar(current, self.home, self.obstacles[-1])
        # if back_path:
        #     all_path += back_path[1:] # Avoid duplicate pos

        return all_path
    
    def plan_path(self):
        """Execute the path planning algorithm to visit all checkpoints"""
        if self.checkpoints:
            self.sort_checkpoints_by_start()
            self.path = self.run_to_checkpoint_and_back()
    
    def sort_checkpoints_by_start(self):
        """Sort the checkpoints that need to be visited """
        if self.start_pos is None:
            return

        current = self.start_pos
        checkpoints = self.checkpoints.copy()
        ordered = []

        while checkpoints:
            nearest = min(checkpoints, key=lambda cp: self.heuristic(current, cp))
            ordered.append(nearest)
            checkpoints.remove(nearest)
            current = nearest

        self.checkpoints = ordered
        return self.checkpoints
    
    def reset_simulator(self):
        """Reset the simulator to initial state"""
        self.obstacles = []
        self.checkpoints = []
        self.path = None
        self.init(0, 0, 0, 0, 0, 0) 

if __name__ == "__main__":
    sim = Simulator()
    # sim.init(0, 0, 0, 0, 0, 0)

    # sim.add_checkpoint(0.5, 0.2)
    # sim.add_checkpoint(1.0, -0.2)
    # sim.add_checkpoint(1.5, 0.3)
    # sim.add_obstacle((1,-1), 0.1)
    # sim.path = sim.run_to_checkpoint_and_back()

    # while True:
    #     sim.step(0.1, 0, 0.1)
    #     sim.render()
    #     time.sleep(0.5)

    running = True
    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                sim.handle_click(event.pos, event.button)

        sim.render()