import csv
from maze import MazeGenerator
from dfs import solve_maze_dfs
from bfs import solve_maze_bfs
from astar import solve_maze_astar

def run_experiments_search(num_runs, maze_sizes, csv_filename="maze_algorithms_results.csv"):
    """
    For every maze size, perform several trials (num_runs).
    Every trial generates a new maze, conducts DFS, BFS, and A*.
    and saves the results (runtime, states expanded, max memory usage, path length) to CSV.
        """
    fieldnames = [
        'algorithm', 'maze_rows', 'maze_cols', 'run',
        'runtime', 'states_expanded', 'peak_memory_usage', 'path_length'
    ]
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for rows, cols in maze_sizes:
            for run in range(1, num_runs + 1):
                print(f"\n--- Maze Size {rows}x{cols}, Run {run} ---")
                # Generate a new maze instance
                maze_gen = MazeGenerator(rows, cols)
                maze_gen.generate_maze()
                maze_gen.add_loops(probability=0.1)

                # 1) DFS metrics
                path_dfs, metrics_dfs = solve_maze_dfs(maze_gen)
                writer.writerow({
                    'algorithm': 'DFS',
                    'maze_rows': rows,
                    'maze_cols': cols,
                    'run': run,
                    'runtime': metrics_dfs.get('runtime_dfs', 0),
                    'states_expanded': metrics_dfs.get('states_expanded_dfs', 0),
                    'peak_memory_usage': metrics_dfs.get('peak_memory_usage_dfs', 0),
                    'path_length': metrics_dfs.get('path_length_dfs', 0)
                })

                # 2) BFS metrics
                path_bfs, metrics_bfs = solve_maze_bfs(maze_gen)
                writer.writerow({
                    'algorithm': 'BFS',
                    'maze_rows': rows,
                    'maze_cols': cols,
                    'run': run,
                    'runtime': metrics_bfs.get('runtime_bfs', 0),
                    'states_expanded': metrics_bfs.get('states_expanded_bfs', 0),
                    'peak_memory_usage': metrics_bfs.get('peak_memory_usage_bfs', 0),
                    'path_length': metrics_bfs.get('path_bfs_length', 0)
                })

                # 3) A* metrics
                path_astar, metrics_astar = solve_maze_astar(maze_gen)
                writer.writerow({
                    'algorithm': 'A*',
                    'maze_rows': rows,
                    'maze_cols': cols,
                    'run': run,
                    'runtime': metrics_astar.get('runtime_astar', 0),
                    'states_expanded': metrics_astar.get('states_expanded_astar', 0),
                    'peak_memory_usage': metrics_astar.get('peak_memory_usage_astar', 0),
                    'path_length': metrics_astar.get('path_length_astar', 0)
                })

                print(f"Finished run {run} for maze size {rows}x{cols}.")

def main():
    # Define different maze sizes (rows, cols)
    maze_sizes = [(10, 10), (30, 30), (50, 50), (100, 100), (150, 150), (200, 200)]
     # Number of runs per maze size
    num_runs = 3 
    csv_filename = "search_algorithms_results.csv"
    run_experiments_search(num_runs, maze_sizes, csv_filename)
    print(f"\nExperiment results saved to {csv_filename}")

if __name__ == '__main__':
    main()