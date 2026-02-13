import tensorflow
from tensorflow.python.ops.linalg_ops import eye


def add(a, b):
    return tensorflow.add(a, b)


a = tensorflow.eye(2, 3)
assert a.shape == (2, 3)
assert a.dtype == tensorflow.float32
b = eye(2, 3)
assert b.shape == (2, 3)
assert b.dtype == tensorflow.float32
c = add(a, b)
