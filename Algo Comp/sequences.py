import numpy as np
import time

start_time = time.time()

# first_line = input()
def n_gen(inputs):
    inputs = inputs.split(' ')
    for i in inputs:
        yield int(i)
g = n_gen(input())
n = next(g)
m = next(g)
k = next(g)
values = []
# n = how many functions are there
# m = maximum length of function combination. 
# so if n = 3 but m = 2, then we can only do (1, 2), (1, 3), (2, 3)
# the sum of the m functions should be divisible by k
def j_gen(a, b, c):
    for v in np.apply_along_axis(lambda j: a*(j**2) + b, 0, np.array(range(1, c + 1))):
        yield v
for i in range(1, n + 1):
    # nth_line = input()
    g = n_gen(input())
    a = next(g)
    b = next(g)
    c = next(g)
    # a, b, c = [int(i) for i in nth_line.split(' ')]
    g = j_gen(a, b, c)
    values.extend([i for i in g])
    # values.extend([a*(j**2) + b for j in range(1, c + 1)])

values = np.array(values) # 1 row
result = values.copy().reshape((-1, 1)) # 1 column
for i in range(1, m):
    result = (values + result).reshape((-1, 1))
count = np.count_nonzero(result % k == 0)
print(count)
# print(values)

end_time = time.time()
print(end_time - start_time, 'seconds')