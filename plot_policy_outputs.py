import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import math
import ast
import seaborn as sns
from matplotlib.animation import FFMpegWriter
import re
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
#from deep_rl.utils.config import Config

import json
import random
import argparse


def _plot_hm_policy_output(data, title, fname):
    data = data.T
    n_actions, n_steps = data.shape

    fig = plt.figure(figsize=(12, 6))
    ax = fig.subplots()
    ax.set_aspect('auto')
    im = ax.imshow(data, cmap='YlGn', vmin=0, vmax=2)  # Updated vmax to 1 for probability scale
    ax.set_yticks(np.arange(n_actions), labels=['A{0}'.format(idx) for idx in range(n_actions)], fontsize=14)
    ax.set_xticks(np.arange(n_steps), labels=['S{0}'.format(idx) for idx in range(n_steps)], fontsize=14)
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(n_actions):
        for j in range(n_steps):
            text = ax.text(j, i, '{0:.2f}'.format(data[i, j]), ha='center', va='center', fontsize=8)
    
    ax.set_title(title, fontsize=16)
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)

def generate_policy_heatmaps(root_dir, max_steps):
    """
    Generates heatmap of action probabilities over steps for each agent by taking the last 32 steps
    of policy logits and calculating softmax probabilities.

    Args:
        root_dir (str): Root directory containing agent directories with logits_data.csv files.
    """
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'logits_data.csv':
                file_path = os.path.join(subdir, file)
                agent_name = os.path.basename(subdir)  # Use subdir name as agent identifier
                
                # Load logits data
                data = pd.read_csv(file_path)

                '''################################################################################
                # Check the total number of rows, excluding the header
                total_rows = data.shape[0]

                # Verify the total number of rows is equal to 500 groups * 32 steps
                expected_rows = 500 * 32
                is_correct_number_of_rows = total_rows == expected_rows
                print("Total rows:", total_rows)
                print("Expected rows (500 groups * 32 steps):", expected_rows)
                print("Correct number of rows:", is_correct_number_of_rows)

                # Check that each group has exactly 32 steps by verifying Step numbers restart at 0 every 32 rows
                group_starts_correctly = all(data['Step'].iloc[i] == 0 for i in range(0, total_rows, 32))
                print("Each group starts with Step 0:", group_starts_correctly)

                # Final check
                if is_correct_number_of_rows and group_starts_correctly:
                    print("The file contains exactly 500 groups of 32 steps each.")
                else:
                    print("The file does NOT contain exactly 500 groups of 32 steps.")

                # Check for rows where the "Step" column starts with 0
                # Assuming the column is named 'Step' exactly as in your data
                rows_start_with_zero = data[data['Step'] == 0]

                # Count the number of rows that start with a Step value of 0
                count_rows_start_with_zero = rows_start_with_zero.shape[0]

                # Display the count
                print(f"Number of rows with Step value of 0: {count_rows_start_with_zero}")
                
                # Identify rows where the "Step" value is 0
                step_zero_indices = data.index[data['Step'] == 0].tolist()

                # Initialize a list to store the count of steps in each group
                step_counts = []

                # Iterate through each group to count the steps
                for i in range(len(step_zero_indices)):
                    # Determine the start and end of the group
                    start_idx = step_zero_indices[i]
                    end_idx = step_zero_indices[i + 1] if i + 1 < len(step_zero_indices) else len(data)
                    
                    # Calculate the number of steps in this group
                    step_count = end_idx - start_idx
                    step_counts.append(step_count)

                # Convert the list to a NumPy array
                step_counts_array = np.array(step_counts)
                
                print('Number of steps in each group:', step_counts_array)

                ################################################################################'''

                # Ensure correct shape: last 32 steps for all actions
                if len(data) < max_steps:
                    print(f"Warning: {file_path} has fewer than 32 steps.")
                    continue  # Skip if insufficient data
                
                # Extract only the last 32 rows of logit data, excluding 'Step' column
                last_32_logits = data.iloc[-max_steps:, 1:].values  # Convert to numpy array, skip 'Step' column
                
                # Convert to PyTorch tensor for softmax calculation
                logits_tensor = torch.tensor(last_32_logits, dtype=torch.float32)
                
                # Calculate softmax probabilities along the action dimension
                softmax_probs = torch.softmax(logits_tensor, dim=1).numpy()  # Convert back to numpy for plotting
                
                # Plot heatmap
                title = f"Action Probabilities Over Steps for {agent_name}"
                output_file = os.path.join(subdir, f"{agent_name}_policy_heatmap.pdf")
                _plot_hm_policy_output(softmax_probs, title, output_file)
                
                print(f"Heatmap generated for {agent_name} at {output_file}")

# Run the function for a specified directory

from matplotlib import animation

def generate_heatmap_video(root_dir, max_steps, interval=200):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'logits_data.csv':
                file_path = os.path.join(subdir, file)
                agent_name = os.path.basename(subdir)  # Use subdir name as agent identifier

                # Load logits data from CSV
                data = pd.read_csv(file_path)
                
                # Ensure data has enough rows to form complete 32-step groups
                n_actions = data.shape[1] - 1  # Excluding 'Step' column
                n_rows = len(data)
                n_groups = n_rows // max_steps  # Calculate the number of 32-step groups

                if n_groups < 1:
                    print(f"Not enough data in {file_path} to form a 32-step group.")
                    return
                
                # Reshape data to (n_groups, 32, n_actions) for easy iteration
                logits_values = data.iloc[:, 1:].values  # Exclude 'Step' column
                logits_tensor = torch.tensor(logits_values, dtype=torch.float32)
                logits_groups = logits_tensor[:n_groups * max_steps].view(n_groups, max_steps, n_actions)

                # Set up the plot for animation
                fig, ax = plt.subplots(figsize=(12, 6))
                cax = ax.imshow(np.zeros((n_actions, max_steps)), cmap="YlGn", vmin=0, vmax=2)  # Empty initial plot
                ax.set_yticks(np.arange(n_actions))
                ax.set_xticks(np.arange(max_steps))
                ax.set_yticklabels([f"A{i+1}" for i in range(n_actions)], fontsize=12)
                ax.set_xticklabels([f"S{j+1}" for j in range(max_steps)], fontsize=12)
                ax.set_title("Policy Action Probabilities Over Time", fontsize=16)

                # Function to update the heatmap at each frame
                def update_heatmap(frame_idx):
                    ax.clear()
                    # Get the logits for the current frame and compute softmax probabilities
                    avg_logits = logits_groups[frame_idx]
                    softmax_probs = torch.softmax(avg_logits, dim=1).numpy()

                    # Update the heatmap plot
                    cax = ax.imshow(softmax_probs.T, cmap="YlGn", vmin=0, vmax=2)
                    ax.set_yticks(np.arange(n_actions))
                    ax.set_xticks(np.arange(max_steps))
                    ax.set_yticklabels([f"A{i+1}" for i in range(n_actions)], fontsize=12)
                    ax.set_xticklabels([f"S{j+1}" for j in range(max_steps)], fontsize=12)
                    ax.set_title(f"Policy Action Probabilities (Group {frame_idx + 1}/{n_groups})", fontsize=16)

                    # Annotate the cells with values
                    for i in range(len(softmax_probs)):
                        for j in range(len(softmax_probs[i])):
                            if j < softmax_probs.shape[1]:  # Ensure 'j' is within bounds
                                ax.text(i, j, f"{softmax_probs[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

                # Create animation by updating the heatmap at each frame
                ani = animation.FuncAnimation(fig, update_heatmap, frames=n_groups, interval=interval)
                
                # Save the animation as an MP4 with H.264 codec
                video_path = os.path.join(subdir, f"{agent_name}_policy_animation.mp4")
                
                # Use ffmpeg writer with extra_args to specify the codec
                ani.save(video_path, writer="ffmpeg", dpi=100, fps=5, extra_args=['-vcodec', 'mpeg4'])

                plt.close(fig)
                print(f"Video saved to {video_path}")

# Example usage
#generate_heatmap_video('log/minihack_debugging/room/fullcomm/run1', 32, 200)

def generate_beta_heatmaps(root_dir, interval):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file == 'betas.csv':
                file_path = os.path.join(subdir, file)
                agent_name = os.path.basename(subdir)  # Use subdir name as agent identifier

                # Load logits data from CSV
                df = pd.read_csv(file_path)

                # Function to clean up the beta column entries
                def clean_betas_column(text):
                    text = re.sub(r"tensor\(\s*\[(.*?)\)", r"[\1]", text, flags=re.DOTALL)      # Remove tensor(...) wrapper from strings
                    text = re.sub(r"\n\s*", " ", text)                                          # Remove \n characters from strings
                    text = re.sub(r", device='cuda:0', grad_fn=<SliceBackward0>]", "", text)    # Remove torch additional tensor formatting data
                    text = re.sub(r", device='cuda:0', grad_fn=<SoftmaxBackward0>]", "", text)
                    return ast.literal_eval(text.strip())                                       # Convert to python dictionary

                # Apply the cleaning function to each entry in the 'betas' column
                df['betas'] = df['betas'].apply(clean_betas_column)
                print(df['betas'][39])

                """# Set up timesteps of interest around each communication cycle
                communication_interval = 20  # Communication cycles occur every 20 timesteps
                timesteps_of_interest = []
                for t in range(communication_interval - 1, len(df), communication_interval):
                    # Filter out-of-bounds indices from timesteps_of_interest
                    timesteps_of_interest.extend([t - 1, t, t + 1])
                timesteps_of_interest = [t for t in timesteps_of_interest if t >= 0 and t < len(df)]

                print(timesteps_of_interest)

                # Extract beta layer names from the first entry
                sample_betas = df['betas'].iloc[0]
                layers = list(sample_betas.keys())

                print(layers)

                # Determine max number of beta parameters for consistent plot sizing
                max_beta_params = max(len(sample_betas[layer]) for layer in layers)

                # Create figure with increased size for larger heatmap cells
                fig, axes = plt.subplots(len(layers), 3, figsize=(20, len(layers) * 6))
                fig.suptitle('Beta Weights Around Communication Cycles', fontsize=18)

                # Loop through each layer and each timestep of interest to plot heatmaps
                for row_idx, layer in enumerate(layers):
                    for col_idx, offset in enumerate([-1, 0, 1]):  # Before, during, after
                        # Get the beta values for the current layer at each relevant timestep
                        beta_values = []
                        for t in timesteps_of_interest[col_idx::3]:  # Adjust indexing for each subset of steps
                            if t in df.index:
                                betas = df.loc[t, 'betas'][layer]
                                beta_values.append(betas)
                        
                        # Ensure beta_values has data; pad for consistent plot dimensions
                        if beta_values:
                            heatmap_data = np.array(beta_values)
                            if heatmap_data.ndim == 1:
                                heatmap_data = np.expand_dims(heatmap_data, axis=0)
                            heatmap_data = np.pad(
                                heatmap_data, 
                                ((0, 0), (0, max_beta_params - heatmap_data.shape[1])), 
                                mode='constant', 
                                constant_values=np.nan
                            )

                            # Plot heatmap with square cells and annotations
                            sns.heatmap(
                                heatmap_data,          # No transpose needed here
                                ax=axes[row_idx, col_idx], 
                                cmap='viridis', 
                                cbar=(col_idx == 2),  # Only show color bar on last column
                                annot=True,           # Add the actual values
                                fmt=".2f",            # Format values to two decimal points
                                annot_kws={"size": 5}, # Font size for annotations
                                square=True            # Make cells square-shaped

                            )
                            
                            # Set titles and labels
                            axes[row_idx, col_idx].set_title(f'{layer} - {"Before" if offset == -1 else "During" if offset == 0 else "After"}')
                            axes[row_idx, col_idx].set_ylabel('Cycle')  # Y-axis for cycles
                            axes[row_idx, col_idx].set_xlabel('Beta Parameter Index')  # X-axis for beta parameter index

                # Adjust layout and save as PDF
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for main title spacing
                fig.savefig("beta_weights_communication_cycles.pdf")
                plt.show()"""


                # Set up timesteps of interest around each communication cycle
                communication_interval = interval  # Communication cycles occur every 20 timesteps
                timesteps_of_interest = []
                for t in range(communication_interval, len(df), communication_interval):
                    # Filter out-of-bounds indices from timesteps_of_interest
                    timesteps_of_interest.extend([t - 1, t, t + 1])
                timesteps_of_interest = [t for t in timesteps_of_interest if t >= 0 and t < len(df)]

                print(timesteps_of_interest)

                # Extract beta layer names from the first entry
                sample_betas = df['betas'].iloc[0]
                layers = list(sample_betas.keys())

                print(layers)

                # Determine max number of beta parameters for consistent plot sizing
                max_beta_params = max(len(sample_betas[layer]) for layer in layers)

                # Create figure
                fig, axes = plt.subplots(len(layers), 3, figsize=(14, len(layers) * 6))
                fig.suptitle('Beta Weights Around Communication Cycles', fontsize=16)

                # Loop through each layer and each timestep of interest to plot heatmaps
                for row_idx, layer in enumerate(layers):
                    for col_idx, offset in enumerate([-1, 0, 1]):  # Before, during, after
                        # Get the beta values for the current layer at each relevant timestep
                        beta_values = []
                        selected_timesteps = []  # Track actual timesteps for y-axis labels
                        
                        for idx, t in enumerate(timesteps_of_interest[col_idx::3]):  # Adjust indexing for each subset of steps
                            betas = df.loc[t, 'betas'][layer]
                            beta_values.append(betas)
                            selected_timesteps.append(t)  # Store actual timestep

                        # Pad beta values to ensure consistent heatmap size
                        heatmap_data = np.array(beta_values)
                        heatmap_data = np.pad(
                            heatmap_data, 
                            ((0, 0), (0, max_beta_params - heatmap_data.shape[1])), 
                            mode='constant', 
                            constant_values=np.nan
                        )
                        
                        # Plot heatmap with no color bar and x-axis values enabled
                        cax = sns.heatmap(
                            heatmap_data, 
                            ax=axes[row_idx, col_idx], 
                            cmap='YlGn', 
                            cbar=False,                  # Remove the color bar
                            vmin=0, vmax=2,
                            xticklabels=True,             # Enable x-axis labels
                            yticklabels=selected_timesteps,  # Set actual timesteps as y-axis labels
                            square=True                    # Keep cells square
                        )
                        
                        # Set titles and labels
                        axes[row_idx, col_idx].set_title(f'{layer} - {"Before" if offset == -1 else "During" if offset == 0 else "After"}')
                        axes[row_idx, col_idx].set_xlabel('Beta Parameter Index')
                        axes[row_idx, col_idx].set_ylabel('Cycle (Timesteps)')

                        # Add text annotations inside the heatmap boxes to show the beta values
                        for i in range(heatmap_data.shape[1]):
                            for j in range(heatmap_data.shape[0]):
                                if not np.isnan(heatmap_data[j, i]):  # Skip NaN values
                                    axes[row_idx, col_idx].text(i + 0.5, j + 0.5, f'{heatmap_data[j, i]:.2f}',
                                                                ha='center', va='center', color='black', fontsize=5)

                # Adjust layout and save as PDF
                plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for main title spacing
                image_path = os.path.join(subdir, f"{agent_name}beta_weights_communication_cycles.pdf")
                fig.savefig(image_path, dpi=300, bbox_inches='tight')





def generate_videos(root_dir):
    """
    Walk through directories to find relevant monitor_log.csv and shell_config.json files,
    match them, and generate videos for each agent.
    """
    for subdir, _, files in os.walk(root_dir):
        monitor_log_path = None
        shell_config_path = None

        # Locate monitor_log.csv and shell_config.json in the directory
        for file in files:
            if 'monitor_log.csv' in file:
                monitor_log_path = os.path.join(subdir, file)

                # Load monitor_log.csv and filter valid rows
                df = pd.read_csv(monitor_log_path)

                # Check for empty or invalid monitor logs
                if df.empty or (df.columns.tolist() == ['episode', 'step', 'action', 'observation', 'reward', 'done', 'truncated', 'info'] and len(df) == 0):
                    print(f"Skipping empty or header-only monitor log: {monitor_log_path}")
                    continue

                break

        for file in files:
            if file == 'shell_config.json':
                shell_config_path = os.path.join(subdir, file)

        if not monitor_log_path or not shell_config_path:
            continue  # Skip if either file is missing

        # Load the shell_config.json
        with open(shell_config_path, 'r') as f:
            shell_config = json.load(f)

        # Extract curriculum details
        curriculum = shell_config.get("curriculum", {})
        task_ids = curriculum.get("task_ids", [])
        max_steps = curriculum.get("max_steps", None)

        if not task_ids or max_steps is None:
            print(f"Skipping {subdir} due to incomplete curriculum in shell_config.json")
            continue

        # Generate the video for this agent
        agent_name = os.path.basename(subdir)
        output_video_file = os.path.join(subdir, f"{agent_name}_replay.mp4")
        print(f"Generating video for agent: {agent_name} using log file: {monitor_log_path}")
        generate_video_for_agent(shell_config, df, output_video_file, task_ids)


def generate_video_for_agent(shell_config, log_data, output_video_file, task_ids):
    """
    Replay actions for an agent and record a video.

    Args:
        shell_config (dict): Parsed shell_config.json for environment setup.
        log_data (pd.DataFrame): DataFrame containing step-level log data.
        output_video_file (str): Path to save the generated video.
    """
    # Create the environment
    env_name = shell_config['env']['env_name']
    env_config_path = shell_config['env']['env_config_path']
    env = create_environment(env_name, env_config_path, task_ids)
    recorder = VideoRecorder(env, output_video_file, enabled=True)

    current_episode = -1
    done = None
    for _, row in log_data.iterrows():
        # Check if we need to reset the environment
        if done or row['episode'] != current_episode:
            observation = env.reset()
            current_episode = row['episode']
            print(f"Starting Episode {current_episode}")

        # Extract action and perform a step
        action = eval(row['action']) if isinstance(row['action'], str) else row['action']
        if not math.isnan(action):
            if isinstance(action, float):
                action = int(action)
            observation, reward, done, info = env.step(action)
        else:
            continue

        # Render and record the frame
        env.render()
        recorder.capture_frame()

        # Break if the episode ends
        if row['done']:
            print(f"Episode {current_episode} finished.")
            continue

    # Close environment and video recorder
    env.close()
    recorder.close()


def create_environment(env_name, env_config_path, task_ids):
    """
    Create the environment based on its name and configuration.

    Args:
        env_name (str): Environment name (e.g., 'minihack').
        env_config_path (str): Path to the environment configuration.

    Returns:
        gym.Env: The environment instance.
    """
    if env_name == 'minihack':
        return minihack(env_config_path, task_ids)
    elif env_name == 'minigrid':
        return minigrid(env_config_path, task_ids)
    elif env_name == 'composuite':
        return composuite(env_config_path, task_ids)
    elif env_name == 'ctgraph':
        return mctgraph(env_config_path, task_ids)
    else:
        raise ValueError(f"Unsupported environment: {env_name}")


# Example environment creation functions
def minihack(env_config_path, task_ids):
    import minihack
    from nle import nethack

    with open(env_config_path, 'r') as f:
        env_config = json.load(f)

    env_names = env_config['tasks'][task_ids[0]]

    MOVE_ACTIONS = tuple(nethack.CompassDirection)
    NAVIGATE_ACTIONS = MOVE_ACTIONS + (
        nethack.Command.OPEN, 
        nethack.Command.KICK
    )

    env = gym.make(env_names, observation_keys=('pixel_crop',), actions=NAVIGATE_ACTIONS)
    return env


def minigrid(env_config_path, task_ids):
    # Example placeholder for Minigrid setup
    return None


def composuite(env_config_path, task_ids):
    # Example placeholder for Composuite setup
    return None


def mctgraph(env_config_path, task_ids):
    # Example placeholder for MCTGraph setup
    return None
    


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='', type=str)
    parser.add_argument('--interval', help='', type=int)
    parser.add_argument('--betas', '--b', '-b', help='plots betas', action='store_true')
    parser.add_argument('--policy', '--p', '-p', help='plots policy', action='store_true')
    parser.add_argument('--video', '--v', '-v', help='plots videos', action='store_true')
    parser.add_argument('--curriculum_id', '--c', '-c', help='curriculum identifier', type=int)
    parser.add_argument('--shell_config_path', '--spath', '-sp', help='shell configuration path for environment', type=str)
    
    
    args = parser.parse_args()

    if args.policy:
        generate_policy_heatmaps(args.path, 32)
    
    if args.betas:
        generate_beta_heatmaps(args.path, args.interval)

    if args.video:
        generate_videos(args.path)
