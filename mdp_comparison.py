import time
import csv
import matplotlib.pyplot as plt

# Import from our separate modules
from maze import MazeGenerator, extract_path
from value_iteration import solve_maze_value_iteration
from policy_iteration import solve_maze_policy_iteration

def main():
    """
    Compare MDP Value Iteration and Policy Iteration for various sizes of mazes.
    Saves results to 'mdp_comparison_results.csv' and prints an example maze run.
    """
    # 1. Maze sizes and runs
    maze_sizes = [(10, 10), (30, 30), (50, 50),(150, 150),(200, 200)]
    # Number of runs per maze size
    num_runs = 3  
    
    results = []  

    for rows, cols in maze_sizes:
        for run in range(1, num_runs + 1):
            print(f"\n--- Maze Size {rows}x{cols}, Run {run} ---")
            # Create a new maze
            generator = MazeGenerator(rows, cols)
            generator.generate_maze()
            generator.add_loops(probability=0.1)

            # Ensure there's a path from start to goal.
            attempts = 0
            while not generator.is_path_to_goal() and attempts < 10:
                generator.add_loops(probability=0.2)
                attempts += 1

            if not generator.is_path_to_goal():
                print(f"Warning: Maze {rows}x{cols} run {run} is not solvable. Skipping.")
                continue

            # 2A. MDP Value Iteration
            start_time = time.time()
            V_vi, policy_vi, states_expanded_vi = solve_maze_value_iteration(
                generator, gamma=0.9, theta=1e-4, max_iter=5000
            )
            runtime_vi = time.time() - start_time

            # Extract solution path and measure length
            policy_dict_vi = {(i, j): policy_vi[(i, j)] for i in range(generator.rows) for j in range(generator.cols)}
            solution_vi = extract_path(policy_dict_vi, generator.start, generator.goal)

            # Peak memory usage is a placeholder => total maze cells
            peak_memory_vi = rows * cols

            # Record metrics for Value Iteration
            results.append({
                "algorithm": "Value Iteration",
                "maze_size": f"{rows}x{cols}",
                "run": run,
                "runtime": runtime_vi,
                "states_expanded": states_expanded_vi,
                "peak_memory": peak_memory_vi,
                "solution_length": len(solution_vi)
            })

            # 2B. MDP Policy Iteration
            start_time = time.time()
            V_pi, policy_pi, states_expanded_pi = solve_maze_policy_iteration(
                generator, gamma=0.9, theta=1e-4
            )
            runtime_pi = time.time() - start_time

            # Extract solution path and measure length
            policy_dict_pi = {(i, j): policy_pi[(i, j)] for i in range(generator.rows) for j in range(generator.cols)}
            solution_pi = extract_path(policy_dict_pi, generator.start, generator.goal)

            # Peak memory usage is a placeholder => total maze cells
            peak_memory_pi = rows * cols

            # Record metrics for Policy Iteration
            results.append({
                "algorithm": "Policy Iteration",
                "maze_size": f"{rows}x{cols}",
                "run": run,
                "runtime": runtime_pi,
                "states_expanded": states_expanded_pi,
                "peak_memory": peak_memory_pi,
                "solution_length": len(solution_pi)
            })

    # 3. Save results to CSV
    csv_filename = "mdp_algorithm_results.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        fieldnames = [
            "algorithm", "maze_size", "run",
            "runtime", "states_expanded", "peak_memory", "solution_length"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\nExperiment results saved to {csv_filename}")

if __name__ == "__main__":
    main()