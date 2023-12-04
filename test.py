import numpy as np

x = [[0, 1, 2], [0, 1, 2]]

mappings = {
    0: [255, 0, 0],
    1: [0, 255, 0],
    2: [0, 0, 255]
}

mapped = np.array([[mappings.get(value) for value in row] for row in x])

print(mapped)
