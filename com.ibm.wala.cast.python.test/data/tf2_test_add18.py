import tensorflow as tensors
from tensorflow.python.ops.variables import Variable


def add(a, b):
    return a + b


v1 = tensors.Variable([1.0, 2.0])
assert v1.shape.as_list() == [2]
assert v1.dtype == tensors.float32

v2 = Variable([2.0, 2.0])
assert v2.shape.as_list() == [2]
assert v2.dtype == tensors.float32

c = add(v1, v2)
