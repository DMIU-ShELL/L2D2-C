y = dict()

y[0] = 'label'
print(y)

_y = dict()
_y[1] = 'label'
_y.update(y)
y = _y

print(y)

y[2] = 'label'
print(y)