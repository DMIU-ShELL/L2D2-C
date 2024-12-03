import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f1(x):
    return 0.5 + 0.5*x

def f2(x, c):
    return (0.5 * (1-x)) / c

# Generate x values
x = np.linspace(-10, 1, 1000)  # 1000 points from -10 to 10
# Calculate y values
y = f1(x)

# Normalize y values only for x in the range [-10, 0]
# Mask to identify the indices in the range [-10, 0]
"""mask = (x >= -10) & (x <= 0)
y_min, y_max = y[mask].min(), y[mask].max()
y[mask] = (y[mask] - y_min) / (y_max - y_min)  # Apply normalization only in the range [-10, 0]
"""


functions = {
    '0.5+0.5x' : y
}


# Create the plot
plt.figure(figsize=(10, 6))
for idx, val in functions.items():
    plt.plot(x, val, label=f'{idx}', linewidth=2)

# Add title and labels
plt.title("Plot of function", fontsize=16)
plt.xlabel("x", fontsize=14)
plt.ylabel("f(x)", fontsize=14)

# Add a legend
plt.legend(fontsize=12)

# Show grid
plt.grid(True)

# Show the plot
plt.savefig('function.pdf', dpi=300)