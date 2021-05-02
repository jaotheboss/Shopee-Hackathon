import numpy as np
import time

start_time = time.time()

first_line = input()
def n_gen(inputs):
    inputs = inputs.split(' ')
    for i in inputs:
        yield int(i)
g = n_gen(first_line)
x = next(g)
y = next(g)

matrix = []
for i in range(x):
       g = n_gen(input())