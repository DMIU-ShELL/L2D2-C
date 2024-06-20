import numpy as np
import pandas as pd
import torch

import numpy as np

def ks_test_change_point(distances, window_size):
  """
  Implements Kolmogorov-Smirnov test for change point detection in distances.

  Args:
      distances: A numpy array of shape (n_samples,) representing the distance history.
      window_size: The size of the window to compare before and after a potential change point.

  Returns:
      A list of indices where potential change points are detected.
  """
  change_points = []
  n_samples = len(distances)

  for i in range(window_size, n_samples):
    # Divide data into before and after windows
    data_before = distances[i - window_size:i]
    data_after = distances[i:]

    # Empirical distribution functions (EDFs)
    cdf_before = np.sort(data_before) / (window_size)
    cdf_after = np.sort(data_after) / len(data_after)

    # Maximum absolute difference (KS statistic)
    D = np.max(np.abs(cdf_before - cdf_after[-1]))

    # Reference critical value (assuming significance level alpha=0.05)
    # You might need to adjust alpha based on your needs
    alpha = 0.05
    critical_value = np.sqrt((window_size + len(data_after)) / (window_size * len(data_after))) * np.sqrt( - np.log(alpha / 2))

    # Check if KS statistic exceeds critical value
    if D > critical_value:
      change_points.append(i)

  return change_points




# Example usage (assuming embeddings and distances are pre-computed)
df = pd.read_csv('log/randproj_distance_test/mctgraph/MetaCTgraph-shell-dist-upz-seed-9177/agent_20/240523-142417/distances.csv')
distances = df['distance'].to_numpy()
distance_history = []
iteration = 0
window_size = 1
for i, distance in enumerate(distances):
    
    
    distance_history.append(distance)

    change_points = ks_test_change_point(np.array(distance_history), window_size)


    if any(change_points):
        print(f"Iteration {iteration} Detected Change Points: {len(change_points)}")
        distance_history = [distance_history[-1]]

    iteration +=1