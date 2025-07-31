from maze import MazeGenerator
import heapq
import time
import argparse

def manhattan_distance(c1, c2):

    # Heuristic function for a star - Manhattan distance between two cells c1=(r1,c1) and c2=(r2,c2).
    
    return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

def solve_maze_astar(maze_gen):
    """
    Solve the maze using the A* algorithm with four metrics_astar:
      1. runtime_astar (seconds)
      2. states_expanded_astar (cells popped from the priority queue)
      3. peak_memory_usage_astar (max queue size)
      4. path_length_astar (length of the found path)
    """
    start_time_astar = time.time()

    start = maze_gen.start
    goal = maze_gen.goal

    # Edge case checks
    if not (0 <= start[0] < maze_gen.rows and 0 <= start[1] < maze_gen.cols):
        return [], {
            "runtime_astar": 0,
            "states_expanded_astar": 0,
            "peak_memory_usage_astar": 0,
            "path_length_astar": 0
        }
    if not (0 <= goal[0] < maze_gen.rows and 0 <= goal[1] < maze_gen.cols):
        return [], {
            "runtime_astar": 0,
            "states_expanded_astar": 0,
            "peak_memory_usage_astar": 0,
            "path_length_astar": 0
        }
    if maze_gen.maze[start] == 1 or maze_gen.maze[goal] == 1:
        return [], {
            "runtime_astar": 0,
            "states_expanded_astar": 0,
            "peak_memory_usage_astar": 0,
            "path_length_astar": 0
        }

    # Priority queue holds (f_score, cell)
    open_set_astar = []
    heapq.heappush(open_set_astar, (0, start))

    came_from_astar = {start: None}
    # cost from start to current
    g_score = {start: 0}  
    states_expanded_astar = 0
    peak_memory_usage_astar = 1  

    while open_set_astar:
        f_current, current = heapq.heappop(open_set_astar)
        states_expanded_astar += 1

        if current == goal:
            break  

        for neighbor in maze_gen.get_neighbors(*current):
            # cost is 1 per move
            tentative_g = g_score[current] + 1 
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                f_score = tentative_g + manhattan_distance(neighbor, goal)
                heapq.heappush(open_set_astar, (f_score, neighbor))
                came_from_astar[neighbor] = current

        if len(open_set_astar) > peak_memory_usage_astar:
            peak_memory_usage_astar = len(open_set_astar)

    # Reconstruct the path
    path = maze_gen.reconstruct_solution_path(came_from_astar, start, goal)
    path_length_astar = len(path)
    runtime_astar = time.time() - start_time_astar

    metrics_astar = {
        "runtime_astar": runtime_astar,
        "states_expanded_astar": states_expanded_astar,
        "peak_memory_usage_astar": peak_memory_usage_astar,
        "path_length_astar": path_length_astar
    }
    return path, metrics_astar

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate and solve a maze using A*."
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
    path, metrics_astar = solve_maze_astar(maze_gen)
    print("A* metrics:", metrics_astar)
    maze_gen.visualize_maze(solution=path, title="Maze with A* Solution")