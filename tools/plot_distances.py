import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Function to process data and generate graph
def process_data(csv_file, title, iterations):
    # Load CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Plotting
    plt.figure(figsize=(500, 8))  # Adjust width and height as needed
    plt.plot(df['iteration'], df['distance'], marker='o', linestyle='-', label='Covariance')

    # Plot vertical lines every 200 iterations
    for i in range(iterations, int(df['iteration'].max()), iterations):
        plt.axvline(x=i, color='red', linestyle='--', alpha=0.5)

    # Annotate each data point with its corresponding value
    for x, y in zip(df['iteration'], df['distance']):
        plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title(title)
    plt.xlabel('Iterations')
    plt.ylabel('Distance')
    plt.grid(True)
    plt.legend()

    # Extract the directory name
    directory_name = os.path.dirname(csv_file)

    # Save the graph as a PDF with an aptly named file
    plt.savefig(os.path.join(directory_name, f'{title.lower()}_graph.pdf'), dpi=256, format='pdf', bbox_inches='tight')
    plt.close()  # Close the plot to avoid overlapping plots

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('paths', help='index of the curriculum to use from the shell config json', type=str, default='log')
    parser.add_argument('--iterations', '--i', '-i', help='iterations per task in experiment', type=int, default=200)

    args = parser.parse_args()

    # Define the root directory
    root_directory = args.paths

    # Iterate through subdirectories
    for subdir, dirs, files in os.walk(root_directory):
        for file in files:
            if file == 'distances.csv':
                # Process distances data and generate graph
                process_data(os.path.join(subdir, file), 'Distances', args.iterations)
            elif file == 'maha_cov_mean.csv':
                # Process covariance data for mean and generate graph
                process_data(os.path.join(subdir, file), 'Mahalanobis Covariance (Mean)', args.iterations)
            elif file == 'maha_cov_ident.csv':
                # Process covariance data for identity and generate graph
                process_data(os.path.join(subdir, file), 'Mahalanobis Covariance (Identity)', args.iterations)
            elif file == 'cos_sim.csv':
                # Process covariance data for identity and generate graph
                process_data(os.path.join(subdir, file), 'Cosine Similarity', args.iterations)
            elif file == 'density.csv':
                # Process covariance data for identity and generate graph
                process_data(os.path.join(subdir, file), 'Density', args.iterations)
            elif file == 'emd.csv':
                # Process covariance data for identity and generate graph
                process_data(os.path.join(subdir, file), 'Wasserstein Distance', args.iterations)
            elif file == 'wdist_log.csv':
                # Process covariance data for identity and generate graph
                process_data(os.path.join(subdir, file), 'Wasserstein Distance from Reference', args.iterations)