import numpy as np


x = np.array([0., 0., 1.])
print(x)

y = tuple(x)
print(y)

z = {}

z[y] = 0.999999

print(z)


a = np.asarray(y)
print(a)

if np.array_equal(x, a):
    print(True)
else:
    print(False)