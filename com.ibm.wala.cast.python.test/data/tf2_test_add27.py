import tensorflow
from tensorflow.python.ops.array_ops import zeros


def add(a, b):
    return a + b


arg = tensorflow.zeros([1, 2])
assert arg.shape == (1, 2)
assert arg.dtype == tensorflow.float32

c = add(arg, zeros([2, 2]))
