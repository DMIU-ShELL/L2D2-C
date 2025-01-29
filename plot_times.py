import matplotlib.pyplot as plt
import numpy as np

# Data
locations = [
    "Austria", "Britain", "Hungary", "Belgian", "Zandvoort", "Monza",
    "Azerbaijan", "Singapore", "COTA", "Mexico", "Brazil", "Vegas", "Qatar", "Abu Dhabi"
]
timings_1 = [
    "01:06.771", "01:28.457", "01:16.489", "01:44.897", "01:11.515", 
    "01:22.275", "01:41.701", "01:30.054", "01:32.616", "01:14.841", 
    "01:09.709", "01:32.524", "01:28.124", "01:23.611"
]
timings_2 = [
    "01:07.750", "01:28.859", "01:17.827", "01:44.768", "01:11.561", 
    "01:21.066", "01:40.700", "01:32.208", "01:33.146", "01:15.043", 
    "01:09.229", "01:32.564", "01:23.365", "01:23.765"
]
differences = [
    -0.979, -0.402, -1.338, 0.129, -0.046, 1.209, 1.001, -2.154,
    -0.530, -0.202, 0.480, -0.040, 4.759, -0.154
]

# Identify the index of maximum and minimum differences by absolute value
abs_differences = np.abs(differences)
max_diff_idx = np.argmax(abs_differences)
min_diff_idx = np.argmin(abs_differences)

# X-axis positions for the locations
x = np.arange(len(locations))

# Colors
positive_color = "#E33122"  # Red
negative_color = "#2C5A78"  # Blue
alpha = 0.5  # 50% transparency

# Create figure and axis
fig, ax = plt.subplots(figsize=(14, 8))

# Plot the positive and negative line segments with corresponding colors
for i in range(len(differences) - 1):
    x_segment = x[i:i+2]
    y_segment = differences[i:i+2]
    if y_segment[0] > 0 and y_segment[1] > 0:  # Positive segment
        ax.plot(x_segment, y_segment, color=positive_color, linewidth=2)
    elif y_segment[0] < 0 and y_segment[1] < 0:  # Negative segment
        ax.plot(x_segment, y_segment, color=negative_color, linewidth=2)
    else:  # Transition through zero
        # Interpolate the zero-crossing point
        zero_x = x[i] + (0 - y_segment[0]) / (y_segment[1] - y_segment[0]) * (x[i+1] - x[i])
        ax.plot([x[i], zero_x], [y_segment[0], 0], color=negative_color if y_segment[0] < 0 else positive_color, linewidth=2)
        ax.plot([zero_x, x[i+1]], [0, y_segment[1]], color=positive_color if y_segment[1] > 0 else negative_color, linewidth=2)

# Shade positive differences
ax.fill_between(x, 0, differences, where=np.array(differences) > 0, 
                color=positive_color, alpha=alpha, interpolate=True, label="Andrea (Ferrari)", linewidth=3)

# Shade negative differences
ax.fill_between(x, 0, differences, where=np.array(differences) < 0, 
                color=negative_color, alpha=alpha, interpolate=True, label="Saptarshi (Red Bull)", linewidth=3)

# Customize x-axis
ax.set_xticks(x)
ax.set_xticklabels(locations, rotation=0, ha="center")
ax.set_xlabel("Location", fontsize=12)

# Customize y-axis
ax.set_ylabel("Time Difference (s)", fontsize=12)
ax.axhline(0, color="black", linewidth=0.8, linestyle="--")  # Add a horizontal line at y=0

# Add grid, legend, and title
ax.grid(alpha=0.3)
ax.legend()
ax.set_title("Unimatrix Zero 2024 Formula One Championship Timing Results", fontsize=14)

# Create the table data
formatted_differences = [f"{diff:+.3f}" for diff in differences]

# Highlight the maximum and minimum differences in the table
highlighted_timings_1 = [
    timings_1[max_diff_idx] if i == max_diff_idx else (timings_1[min_diff_idx] if i == min_diff_idx else t)
    for i, t in enumerate(timings_1)
]
highlighted_timings_2 = [
    timings_2[max_diff_idx] if i == max_diff_idx else (timings_2[min_diff_idx] if i == min_diff_idx else t)
    for i, t in enumerate(timings_2)
]
highlighted_differences = [
    f"{formatted_differences[max_diff_idx]}" if i == max_diff_idx else (f"{formatted_differences[min_diff_idx]}" if i == min_diff_idx else d)
    for i, d in enumerate(formatted_differences)
]

# Table data for the table
table_data = [
    highlighted_timings_1, 
    highlighted_timings_2, 
    highlighted_differences
]
row_labels = ["Saptarshi", "Andrea", "Delta"]

# Create the table
table = ax.table(cellText=table_data, rowLabels=row_labels, colLabels=locations,
                 cellLoc="center", loc="bottom", bbox=[0.05, -0.4, 0.9, 0.3])

# Style the table
table.auto_set_font_size(False)
table.set_fontsize(10)

# Adjust layout to fit table
plt.tight_layout()

# Save the figure
plt.savefig('timings.pdf', dpi=300, bbox_inches='tight')