import pandas as pd
import matplotlib.pyplot as plt

# Load CSV file into a pandas DataFrame
csv_file = 'log/MetaCTgraph-shell-dist-wasserstein_distances_test_no_moving_avg-seed-9187/agent_30/240212-183157/distances.csv'
df = pd.read_csv(csv_file)

# Convert 'distance' column to floats
df['distance'] = df['distance'].astype(float)

# Plotting
plt.figure(figsize=(500, 8))  # Adjust width and height as needed
plt.plot(df['iteration'], df['distance'], marker='o', linestyle='-', label='All iterations')

# Plot vertical lines every 200 iterations
for i in range(200, int(df['iteration'].max()), 200):
    plt.axvline(x=i, color='red', linestyle='--', alpha=0.5)

# Annotate each data point with its corresponding value
for x, y in zip(df['iteration'], df['distance']):
    plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.title('Distances over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Distances')
plt.grid(True)
plt.legend()

plt.savefig('distances.pdf', dpi=256, format='pdf', bbox_inches='tight')
