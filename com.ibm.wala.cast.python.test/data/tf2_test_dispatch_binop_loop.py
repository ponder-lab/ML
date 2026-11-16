# Test: a chain of binary operators in a loop that carries its first two values, so the values
# form one cycle. The generator dispatch of a value asks for each operand's, and walked every simple
# path through the cycle (wala/ML#1274).
import tensorflow as tf


def chain(t):
    v0 = t
    v1 = t + 1.0
    for _ in range(2):
        v2 = v0 + v1
        v3 = v1 + v2
        v4 = v2 + v3
        v5 = v3 + v4
        v6 = v4 + v5
        v7 = v5 + v6
        v8 = v6 + v7
        v9 = v7 + v8
        v10 = v8 + v9
        v11 = v9 + v10
        v12 = v10 + v11
        v13 = v11 + v12
        v14 = v12 + v13
        v15 = v13 + v14
        v16 = v14 + v15
        v17 = v15 + v16
        v0 = v16 + v17
        v1 = v17 + v0
    return v17


def consume(t):
    return t


r = chain(tf.ones([2, 3]))
assert r.shape == (2, 3), r.shape
assert r.dtype == tf.float32, r.dtype
consume(r)
