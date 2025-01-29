import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Function to process data and generate graph
def process_data(csv_file, title):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Select only the columns we need
    df = df[['iteration', 'task_id', 'ip', 'port']]

    # Convert 'port' column to strings
    df['port'] = df['port'].astype(str)

    # Concatenate IP and Port to form a unique identifier
    df['IP_Port'] = df['ip'] + ':' + df['port']

    # Pivot the DataFrame to create a matrix where rows represent iterations,
    # columns represent task IDs, and the values represent the count of exchanges
    pivot_df = df.pivot_table(index='iteration', columns='task_id', aggfunc='size', fill_value=0)

    # Plotting
    plt.figure(figsize=(12, 200))
    sns.heatmap(pivot_df, cmap='Blues', annot=True, fmt='d', annot_kws={"fontsize": 8}, square=True, cbar=False, linewidths=2)
    #plt.title('Exchanges of Data Between RL Agents Over Time')
    plt.xlabel('Task ID')
    plt.ylabel('Iteration')

    # Extract the directory name
    directory_name = os.path.dirname(csv_file)

    # Save the heatmap as a PDF
    directory_name = os.path.dirname(csv_file)
    plt.savefig(os.path.join(directory_name, f'{title.lower()}_heatmap.pdf'), dpi=256, format='pdf')
    plt.close()  # Close the plot to avoid overlapping plots

    # Plotting cumulative frequency bar chart
    plt.figure(figsize=(12, 8))
    cumulative_freq = df['task_id'].value_counts()

    # Ensure task_id is treated as categorical data and sort them in ascending order
    cumulative_freq = cumulative_freq.reindex(sorted(cumulative_freq.index))

    cumulative_freq.plot(kind='bar', color='skyblue')
    #plt.title('Cumulative Frequency of Task IDs')
    plt.xlabel('Task ID')
    plt.ylabel('Cumulative Frequency')
    plt.xticks(rotation=0)

    for index, value in enumerate(cumulative_freq):
        plt.text(index, value, str(value), ha='center', va='bottom')

    # Save the cumulative frequency bar chart as a PDF
    plt.savefig(os.path.join(directory_name, f'{title.lower()}_cumulative_freq.pdf'), dpi=256, format='pdf', bbox_inches='tight')
    plt.close()  # Close the plot to avoid overlapping plots

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('paths', help='index of the curriculum to use from the shell config json', type=str, default='log')

    args = parser.parse_args()

    # Define the root directory
    root_directory = args.paths

    # Iterate through subdirectories
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file == 'exchanges.csv':
                # Process distances data and generate graph
                process_data(os.path.join(subdir, file), 'Exchanges')