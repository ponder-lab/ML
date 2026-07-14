# Reproducer for wala/ML#729: a def derived by enumerate-iterating a numpy-fed tensor parameter
# must read numpy-only origins, like the map/for-loop contrast cases, since Python iteration is
# not a traceable TensorFlow operation (enumerate over a symbolic tensor raises under tracing).
import numpy as np


def f(nodes):
    return {int(j): i for i, j in enumerate(nodes)}


def g(nodes):
    return list(map(int, nodes))


def h(nodes):
    total = 0
    for n in nodes:
        total += int(n)
    return total


def consume_f(x):
    pass


a = np.array([1, 2, 3])
fa = f(a)
assert fa == {1: 0, 2: 1, 3: 2}
ga = g(a)
assert ga == [1, 2, 3]
ha = h(a)
assert ha == 6
consume_f(a)
