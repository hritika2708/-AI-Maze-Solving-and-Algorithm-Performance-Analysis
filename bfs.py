import collections
import time
import argparse
from maze import MazeGenerator

def solve_maze_bfs(maze_gen):
    """
    Solve the maze using Breadth-First Search (BFS) with four metrics:
      1. runtime_bfs (seconds)
      2. states_expanded_bfs (cells dequeued)
      3. peak_memory_usage_bfs (max queue size)
      4. path_bfs_length (length of the found path)
    """
    start_time_bfs = time.time()

    start = maze_gen.start
    goal = maze_gen.goal

    # Edge case checks
    if not (0 <= start[0] < maze_gen.rows and 0 <= start[1] < maze_gen.cols):
        return [], {
            "runtime_bfs": 0,
            "states_expanded_bfs": 0,
            "peak_memory_usage_bfs": 0,
            "path_bfs_length": 0
        }
    if not (0 <= goal[0] < maze_gen.rows and 0 <= goal[1] < maze_gen.cols):
        return [], {
            "runtime_bfs": 0,
            "states_expanded_bfs": 0,
            "peak_memory_usage_bfs": 0,
            "path_bfs_length": 0
        }
    if maze_gen.maze[start] == 1 or maze_gen.maze[goal] == 1:
        return [], {
            "runtime_bfs": 0,
            "states_expanded_bfs": 0,
            "peak_memory_usage_bfs": 0,
            "path_bfs_length": 0
        }

    queue_bfs = collections.deque([start])
    visited_bfs = set([start])
    came_from_bfs = {start: None}

    states_expanded_bfs = 0
    # queue_bfs starts with 1 item
    peak_memory_usage_bfs = 1  

    # BFS loop
    while queue_bfs:
        current = queue_bfs.popleft()
        states_expanded_bfs += 1

        if current == goal:
            break

        for neighbor in maze_gen.get_neighbors(*current):
            if neighbor not in visited_bfs:
                visited_bfs.add(neighbor)
                came_from_bfs[neighbor] = current
                queue_bfs.append(neighbor)

        if len(queue_bfs) > peak_memory_usage_bfs:
            peak_memory_usage_bfs = len(queue_bfs)

    # Reconstruct path from goal back to start
    path_bfs = maze_gen.reconstruct_solution_path(came_from_bfs, start, goal)
    path_bfs_length = len(path_bfs)

    runtime_bfs = time.time() - start_time_bfs
    metrics = {
        "runtime_bfs": runtime_bfs,
        "states_expanded_bfs": states_expanded_bfs,
        "peak_memory_usage_bfs": peak_memory_usage_bfs,
        "path_bfs_length": path_bfs_length
    }
    return path_bfs, metrics
#setting runtime arguments 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and solve a maze using BFS."
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

    maze_gen = MazeGenerator(args.rows, args.cols)
    maze_gen.generate_maze()
    maze_gen.add_loops(probability=0.1)
    path_bfs, metrics = solve_maze_bfs(maze_gen)
    print("BFS Metrics:", metrics)
    maze_gen.visualize_maze(solution=path_bfs, title="Maze with BFS Solution")