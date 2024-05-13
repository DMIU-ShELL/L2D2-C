import numpy as np
import pandas as pd

def cusum_change_detection(data, threshold=1, drift=0.0):
    n = len(data)
    s = np.zeros(n)     # CUSUM statistic
    change_points = []

    # Initialize the first value
    s[0] = data[0] - drift

    for i in range(1, n):
        # Update CUSUM statistic
        s[i] = max(0, s[i-1] + data[i] - drift)

        if s[i] >= threshold:
            change_points.append(i)

    return change_points


df = pd.read_csv('log/ctgraph_seq_img_seed_sanity_check/MetaCTgraph-shell-dist-upz-seed-6652/agent_4/240411-184231/distances.csv')
data = df['distance'].tolist()


change_points = cusum_change_detection(data, threshold=3, drift=0.1)

print(f'Detected change points: {change_points}')