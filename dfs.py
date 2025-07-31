import time
import argparse
from maze import MazeGenerator

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def solve_maze_dfs(maze_gen):
    """
    Use Depth-First Search (DFS) with four metrics_dfs to solve the maze.
    1. runtime_dfs: total time in seconds
    2. states_expanded_dfs: number of cells popped from the stack_dfs
    3. peak_memory_usage_dfs: maximum stack_dfs size ever existing
    4. path_length_dfs: length of the last path (quality of solution)
    """
    start_time_dfs = time.time()

    start = maze_gen.start
    goal = maze_gen.goal

    # Edge case handling: if start or goal is out of bounds or is a wall, no path is possible
    if not (0 <= start[0] < maze_gen.rows and 0 <= start[1] < maze_gen.cols):
        return [], {
            "runtime_dfs": 0,
            "states_expanded_dfs": 0,
            "peak_memory_usage_dfs": 0,
            "path_length_dfs": 0
        }
    if not (0 <= goal[0] < maze_gen.rows and 0 <= goal[1] < maze_gen.cols):
        return [], {
            "runtime_dfs": 0,
            "states_expanded_dfs": 0,
            "peak_memory_usage_dfs": 0,
            "path_length_dfs": 0
        }
    if maze_gen.maze[start] == 1 or maze_gen.maze[goal] == 1:
        return [], {
            "runtime_dfs": 0,
            "states_expanded_dfs": 0,
            "peak_memory_usage_dfs": 0,
            "path_length_dfs": 0
        }

    stack_dfs = [start]
    visited_dfs = set([start])
    came_from_dfs = {start: None}

    states_expanded_dfs = 0
    # stack_dfs initially has 1 item
    peak_memory_usage_dfs = 1  

    while stack_dfs:
        current = stack_dfs.pop()
        states_expanded_dfs += 1

        if current == goal:
            break

        for neighbor in maze_gen.get_neighbors(*current):
            if neighbor not in visited_dfs:
                visited_dfs.add(neighbor)
                came_from_dfs[neighbor] = current
                stack_dfs.append(neighbor)

        if len(stack_dfs) > peak_memory_usage_dfs:
            peak_memory_usage_dfs = len(stack_dfs)

    # Reconstruct the path
    path = maze_gen.reconstruct_solution_path(came_from_dfs, start, goal)
    path_length_dfs = len(path)

    runtime_dfs = time.time() - start_time_dfs
    metrics_dfs = {
        "runtime_dfs": runtime_dfs,
        "states_expanded_dfs": states_expanded_dfs,
        "peak_memory_usage_dfs": peak_memory_usage_dfs,
        "path_length_dfs": path_length_dfs
    }
    return path, metrics_dfs


# Setting runtime arguments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and solve a maze using DFS."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=50,
        help="Number of rows for the maze (default: 50)"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=50,
        help="Number of columns for the maze (default: 50)"
    )
    args = parser.parse_args()

    rows, cols = args.rows, args.cols

    maze_gen = MazeGenerator(rows, cols)
    maze_gen.generate_maze()
    maze_gen.add_loops(probability=0.1)
    path, metrics_dfs = solve_maze_dfs(maze_gen)
    print("DFS metrics:")
    for metric, value in metrics_dfs.items():
        print(f"  {metric}: {value}")
    maze_gen.visualize_maze(solution=path, title="Maze with DFS Solution")