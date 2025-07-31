import argparse
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import deque

class MazeGenerator:
    def __init__(self, rows, cols):
        """
        Single-array maze of size rows x cols:
          1 -> wall
          0 -> open
        We'll cut out passages with a one-step DFS strategy.
        It holds for odd as well as even dimensions without asking them to be odd.
        Typical start/goal positions are set inside the grid.
        """
        self.rows = rows
        self.cols = cols
        self.maze = np.ones((self.rows, self.cols), dtype=int)

        # Typical start/goal inside the grid
        self.start = (1, 1)
        self.goal = (self.rows - 2, self.cols - 2)

        # Keep track of which cells have been visited during maze generation
        self.visited_maze_gen = np.zeros((self.rows, self.cols), dtype=bool)

    def count_open_neighbors(self, r, c):
        """
        Returns how many of the direct neighbors of (r, c) are open (0).
        """
        count = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            if 0 <= rr < self.rows and 0 <= cc < self.cols:
                if self.maze[rr, cc] == 0:
                    count += 1
        return count

    def generate_maze(self):
        """
        Single-step DFS with '1 open neighbor' test for maze generation:
          - Start at self.start, mark it open (0).
          - Use DFS with a stack.
          - At each step, consider unvisited wall neighbors (inside the inner area)
            that have exactly 1 open neighbor.
          - Open one at random and push it onto the stack.
          - Backtrack if no valid neighbors are left.
        """
        sr, sc = self.start
        self.maze[sr, sc] = 0
        self.visited_maze_gen[sr, sc] = True

        stack_maze = [(sr, sc)]
        while stack_maze:
            r, c = stack_maze[-1]

            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 1 <= nr < self.rows - 1 and 1 <= nc < self.cols - 1:
                    if not self.visited_maze_gen[nr, nc] and self.maze[nr, nc] == 1:
                        if self.count_open_neighbors(nr, nc) == 1:
                            neighbors.append((nr, nc))
            if neighbors:
                nr, nc = random.choice(neighbors)
                self.maze[nr, nc] = 0
                self.visited_maze_gen[nr, nc] = True
                stack_maze.append((nr, nc))
            else:
                stack_maze.pop()

        # Ensure the goal cell is open if it's in range.
        gr, gc = self.goal
        if 0 <= gr < self.rows and 0 <= gc < self.cols:
            self.maze[gr, gc] = 0

    def add_outer_walls(self):
        """
        Make sure the outer boundary is walled off.
        This encloses the maze so paths won't bleed off the edges.
        """
        self.maze[0, :] = 1
        self.maze[-1, :] = 1
        self.maze[:, 0] = 1
        self.maze[:, -1] = 1

    def add_loops(self, probability=0.1):
        """
        Randomly remove some walls to create loops.
        """
        for r in range(1, self.rows - 1):
            for c in range(1, self.cols - 1):
                if self.maze[r, c] == 1 and random.random() < probability:
                    self.maze[r, c] = 0

    def get_neighbors(self, r, c):
        """
        Return valid open neighbors in directions (up, down, left, right) of cell (r, c).
        """
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols:
                # Check for open cell.
                if self.maze[nr, nc] == 0:
                    neighbors.append((nr, nc))
        return neighbors

    def reconstruct_solution_path(self, came_from, start, goal):
        """
        Reconstruct the path from 'start' cell to 'goal' cell using the came_from dictionary.
        """
        path = []
        current = goal
        while current is not None:
            path.append(current)
            current = came_from.get(current, None)
        path.reverse()
        if path and path[0] == start:
            return path
        return []
    
    def is_path_to_goal(self):
        """
        Verify through BFS if there is any path between self.start and self.goal.
        Returns True if within reach, False otherwise.
        """
        queue = deque([self.start])
        visited_maze_gen = set([self.start])
        while queue:
            r, c = queue.popleft()
            if (r, c) == self.goal:
                return True
            for nr, nc in [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]:
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.maze[nr, nc] == 0 and (nr, nc) not in visited_maze_gen:
                        visited_maze_gen.add((nr, nc))
                        queue.append((nr, nc))
        return False
    
    def initialize_values_bfs(self):
        """
        Perform a BFS from the goal and mark all the cells with the value -distance.
        Cells out of reach of the target are -9999.
        """
        dist = np.full((self.rows, self.cols), np.inf)
        queue = deque([self.goal])
        dist[self.goal] = 0
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    if self.maze[nr, nc] == 0 and dist[nr, nc] > dist[r, c] + 1:
                        dist[nr, nc] = dist[r, c] + 1
                        queue.append((nr, nc))
        V_init = np.full((self.rows, self.cols), -9999.0)
        for i in range(self.rows):
            for j in range(self.cols):
                if dist[i, j] < np.inf:
                    V_init[i, j] = -dist[i, j]
        return V_init

    def visualize_maze(self, solution=None, title="Generated Maze"):
        """
        Visualize the maze in Matplotlib:
          - 0 -> white (open)
          - 1 -> black (wall)
          - 2 -> green (start)
          - 3 -> red (goal)
          - 4 -> blue (solution path)
        """
        self.add_outer_walls()

        figsize_maze_gen = (max(6, self.cols / 10), max(6, self.rows / 10))
        fig, ax = plt.subplots(figsize=figsize_maze_gen)

        # Copy the maze for display
        maze_display = np.copy(self.maze)

        # Mark start and goal if in bounds
        sr, sc = self.start
        gr, gc = self.goal
        if 0 <= sr < self.rows and 0 <= sc < self.cols:
            maze_display[sr, sc] = 2
        if 0 <= gr < self.rows and 0 <= gc < self.cols:
            maze_display[gr, gc] = 3

        # If a solution path is provided, mark it in blue (4)
        if solution:
            for (r, c) in solution:
                if (r, c) not in [self.start, self.goal]:
                    maze_display[r, c] = 4

        # Create a custom color map.
        cmap_maze_gen = mcolors.ListedColormap(["white", "black", "green", "red", "blue"])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = mcolors.BoundaryNorm(bounds, cmap_maze_gen.N)

        ax.imshow(maze_display, cmap=cmap_maze_gen, norm=norm)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
        plt.show()


def extract_path(policy, start, goal, max_steps=1000):
    """Extract a path for MDP algorithms by following the policy from start to goal."""
    path_mdp = [start]
    current = start
    for _ in range(max_steps):
        if current == goal:
            break
        a = policy.get(current, '')
        if not a:
            break
        di, dj = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}[a]
        next_state = (current[0] + di, current[1] + dj)
        if next_state == current:
            break
        path_mdp.append(next_state)
        current = next_state
    return path_mdp

if __name__ == '__main__':
    parser_maze_gen = argparse.ArgumentParser(description='Create a maze with rows x cols dimensions.')
    parser_maze_gen.add_argument('rows', type=int, help='Number of rows to be generated')
    parser_maze_gen.add_argument('cols', type=int, help='Number of columns to be generated.')
    args = parser_maze_gen.parse_args()
    
    maze_gen = MazeGenerator(args.rows, args.cols)
    maze_gen.generate_maze()
    maze_gen.add_loops(probability=0.1)
    maze_gen.visualize_maze()