import tensorflow
from tensorflow.python.ops.array_ops import zeros


def add(a, b):
    return a + b


arg = tensorflow.zeros([1, 2], tensorflow.int32)
assert arg.shape == (1, 2)
assert arg.dtype == tensorflow.int32

c = add(arg, zeros([2, 2], tensorflow.int32))
