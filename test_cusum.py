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
        print(data[i], diff, cumulative_sum)

        # Check if the cumulative sum exceeds the threshold
        if cumulative_sum >= threshold:
            change_points.append((i, data[i]))

    return change_points

# Example usage:
#history = [6.42, 5.19, 4.11, 5.48, 3.86, 4.15, 3.87, 3.96, 3.11, 4.16]  # History of the last 10 data points
#new_data_point = 9.26  # Newly generated data point

history = [3.75, 3.69, 4.74, 4.21, 4.46, 3.67, 5.58, 3.85, 1.96, 4.23]
new_data_point = 16.57

# Calculate the difference between the new data point and the mean of the history
mean_history = sum(history) / len(history)
diff = new_data_point - mean_history

# Set a threshold (this value may need to be adjusted based on the data)
threshold = 1 # Example threshold

# Apply CUSUM algorithm
history.append(new_data_point)
change_points = cusum(history, threshold)

print(diff, threshold, change_points)

# Check if the difference exceeds the threshold at any change point
significant_change = any(diff >= threshold for change_point in change_points)

if significant_change:
    print("Significant change detected!")
else:
    print("No significant change detected.")
