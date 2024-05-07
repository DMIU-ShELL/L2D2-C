import pandas as pd
import numpy as np
import time

def cusum(data, threshold):
    """
    CUSUM algorithm for detecting significant changes in a data stream.

    Parameters:
    - data: List or array of data points.
    - threshold: Threshold for detecting significant changes.

    Returns:
    - List of change points where significant changes are detected.
    """
    change_points = []
    cumulative_sum = 0

    for i in range(1, len(data)):
        # Calculate the difference between the current data point and the previous one
        diff = data[i] - data[i - 1]

        # Update the cumulative sum
        cumulative_sum = max(0, cumulative_sum + diff - threshold)
        #print(data[i], diff, cumulative_sum)

        # Check if the cumulative sum exceeds the threshold
        if cumulative_sum >= threshold:
            change_points.append((i, data[i], diff))

    return change_points

def fir_cusum(data, threshold, k):
    """
    Fast Initial Response (FIR) CUSUM algorithm for change point detection.

    Parameters:
    - data: List or array of data points.
    - threshold: Threshold for detecting changes.
    - k: Number of initial observations to use for fast initial response.

    Returns:
    - List of change points where significant changes are detected.
    """
    change_points = []
    s = 0  # Cumulative sum
    s_fast = 0  # Cumulative sum for fast initial response
    n = len(data)
    
    for i in range(n):
        # Calculate the deviation from the mean
        deviation = data[i] - sum(data[:i+1]) / (i + 1)
        
        # Update the cumulative sums
        s = max(0, s + deviation - threshold)
        if i < k:
            s_fast = max(0, s_fast + deviation - threshold)
        
        # Check for change points
        if i >= k and s >= threshold:
            change_points.append(i)
        elif i >= k and i >= 2 * k and s_fast >= threshold:
            change_points.append(i - k)
    
    return change_points



if __name__ == '__main__':
    df = pd.read_csv('log/ctgraph_seq_img_seed_sanity_check/MetaCTgraph-shell-dist-upz-seed-6652/agent_4/240411-184231/distances.csv')
    dataset = df['distance'].tolist()

    history = []
    threshold = 2
    k = 0

    for i, data_point in enumerate(dataset):
        #time.sleep(0.1)
        print(i, data_point)

        if len(history) > 0:
            mean_history = sum(history) / len(history)
            diff = data_point - mean_history

        history.append(data_point)
        change_points = cusum(history, threshold)

        significant_change = any(diff >= threshold for change_point in change_points)

        if significant_change:
            print(f'Significant change detected with change points: {change_points[-1]} diff: {diff} mean_history: {mean_history}')