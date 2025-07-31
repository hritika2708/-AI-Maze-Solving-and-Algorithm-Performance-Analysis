from maze import MazeGenerator, extract_path
import matplotlib.pyplot as plt
import argparse
import time

def solve_maze_policy_iteration(generator, gamma=0.9, theta=1e-4):
    """
    Solve maze with Policy Iteration.
    Returns:
    V: List of value estimates, 2D.
    policy: (row, col) -> action dictionary.
    states_expanded_policy: Overall number of state evaluations.
    """
    actions = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
    # Initialize V as a 2D list and policy as a 2D list with default action 'U'
    V = [[0 for _ in range(generator.cols)] for _ in range(generator.rows)]
    policy_arr = [['U' for _ in range(generator.cols)] for _ in range(generator.rows)]
    for i in range(generator.rows):
        for j in range(generator.cols):
            if generator.maze[i, j] == 1:
                policy_arr[i][j] = ''

    states_expanded_policy = 0
    policy_stable = False
    while not policy_stable:
        # Policy Evaluation
        while True:
            delta = 0
            new_V = [row[:] for row in V]
            for i in range(generator.rows):
                for j in range(generator.cols):
                    if generator.maze[i, j] == 1 or (i, j) == generator.goal:
                        continue
                    states_expanded_policy += 1
                    a = policy_arr[i][j]
                    di, dj = actions[a]
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < generator.rows and 0 <= nj < generator.cols) or generator.maze[ni][nj] == 1:
                        ni, nj = i, j
                    v_new = -1 + gamma * V[ni][nj]
                    new_V[i][j] = v_new
                    delta = max(delta, abs(v_new - V[i][j]))
            V = [row[:] for row in new_V]
            if delta < theta:
                break

        # Policy Improvement
        policy_stable = True
        for i in range(generator.rows):
            for j in range(generator.cols):
                if generator.maze[i, j] == 1 or (i, j) == generator.goal:
                    continue
                states_expanded_policy += 1
                old_action = policy_arr[i][j]
                q_values_policy = {}
                for a, (di, dj) in actions.items():
                    ni, nj = i + di, j + dj
                    if not (0 <= ni < generator.rows and 0 <= nj < generator.cols) or generator.maze[ni][nj] == 1:
                        ni, nj = i, j
                    q_values_policy[a] = -1 + gamma * V[ni][nj]
                best_action = max(q_values_policy, key=q_values_policy.get)
                policy_arr[i][j] = best_action
                if best_action != old_action:
                    policy_stable = False

    policy = {(i, j): policy_arr[i][j] for i in range(generator.rows) for j in range(generator.cols)}
    return V, policy, states_expanded_policy

def main():
    parser = argparse.ArgumentParser(
        description="Generate and solve a maze using MDP Policy Iteration."
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

    # Solve the maze and measure runtime_policy.
    start_time = time.time()
    V, policy, states_expanded_policy = solve_maze_policy_iteration(generator, gamma=0.9, theta=1e-4)
    runtime_policy = time.time() - start_time

    # Extract the solution path.
    policy_dict = {(i, j): policy[(i, j)] for i in range(generator.rows) for j in range(generator.cols)}
    solution = extract_path(policy_dict, generator.start, generator.goal)
    
    # Define and print evaluation metrics.
    peak_memory_policy = generator.rows * generator.cols  # Placeholder for peak memory usage
    
    
    print("Evaluation Metrics:")
    print(f"runtime_policy (seconds): {runtime_policy:.4f}")
    print(f"States Expanded: {states_expanded_policy}")
    print(f"Peak Memory Usage (cells): {peak_memory_policy}")

    print("Solution path length (Policy Iteration):", len(solution))
    generator.visualize_maze(solution=solution, title="MDP Policy Iteration Path")
    plt.show()

if __name__ == "__main__":
    main()