from maze import MazeGenerator, extract_path
import matplotlib.pyplot as plt
import argparse
import time

def solve_maze_value_iteration(generator, gamma=0.9, theta=1e-4, max_iter=5000):
    """
    Solve the maze with Value Iteration.
    Returns:
    V: 2D numpy value estimates array.
    policy: Dict mapping (row, col) -> action.
    states_expanded_value: Number of state evaluations.
    """
    actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    
    V = generator.initialize_values_bfs()
    V[generator.goal] = 0.0
    # Create a 2D array to hold policy actions
    policy_arr_value = [['' for _ in range(generator.cols)] for _ in range(generator.rows)]
    
    states_expanded_value = 0
    for _ in range(max_iter):
        delta = 0
        new_V = V.copy()
        for i in range(generator.rows):
            for j in range(generator.cols):
                # Skip walls and the goal
                if generator.maze[i, j] == 1 or (i, j) == generator.goal:
                    continue
                # Count this state evaluation
                states_expanded_value += 1  
                q_values = {}
                for a, (di, dj) in actions.items():
                    ni, nj = i + di, j + dj
                    if 0 <= ni < generator.rows and 0 <= nj < generator.cols and generator.maze[ni, nj] == 0:
                        val = -1 + gamma * V[ni, nj]
                    else:
                        val = float('-inf')
                    q_values[a] = val
                best_action = max(q_values, key=q_values.get)
                best_value = q_values[best_action]
                if best_value == float('-inf'):
                    new_V[i, j] = V[i, j]
                    policy_arr_value[i][j] = ''
                else:
                    new_V[i, j] = best_value
                    policy_arr_value[i][j] = best_action
                    delta = max(delta, abs(best_value - V[i, j]))
        V = new_V
        if delta < theta:
            break
    
    policy = {(i, j): policy_arr_value[i][j] for i in range(generator.rows) for j in range(generator.cols)}
    return V, policy, states_expanded_value

def main():
    parser = argparse.ArgumentParser(
        description="Generate and solve a maze using MDP Value Iteration."
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=100,
        help="Number of rows for the maze (default: 100)"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=100,
        help="Number of columns for the maze (default: 100)"
    )
    args = parser.parse_args()

    generator = MazeGenerator(args.rows, args.cols)
    generator.generate_maze()
    generator.add_loops(probability=0.1)
    
    # Ensure a path exists from start to goal.
    attempts = 0
    while not generator.is_path_to_goal() and attempts < 10:
        generator.add_loops(probability=0.2)
        attempts += 1
    if not generator.is_path_to_goal():
        print("Warning: Maze is not solvable from start to goal.")
        return

    # Solve the maze and track runtime_value.
    start_time = time.time()
    V, policy, states_expanded_value = solve_maze_value_iteration(generator, gamma=0.9, theta=1e-4, max_iter=5000)
    runtime_value = time.time() - start_time

    # Extract the solution path
    policy_dict = {(i, j): policy[(i, j)] for i in range(generator.rows) for j in range(generator.cols)}
    solution = extract_path(policy_dict, generator.start, generator.goal)
    
    # Define and print evaluation metrics:
    peak_memory_value = generator.rows * generator.cols  # Placeholder metric for peak memory usage
    
   
    print("Evaluation Metrics:")
    print(f"runtime_value (seconds): {runtime_value:.4f}")
    print(f"States Expanded: {states_expanded_value}")
    print(f"Peak Memory Usage (cells): {peak_memory_value}")

    print("Solution path length (Value Iteration):", len(solution))
    generator.visualize_maze(solution=solution, title="MDP Value Iteration Path")
    plt.show()

if __name__ == "__main__":
    main()