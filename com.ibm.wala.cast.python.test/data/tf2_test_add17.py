import tensorflow


def add(a, b):
    return a + b


v1 = tensorflow.Variable([1.0, 2.0])
assert v1.shape.as_list() == [2]
assert v1.dtype == tensorflow.float32

v2 = tensorflow.Variable([2.0, 2.0])
assert v2.shape.as_list() == [2]
assert v2.dtype == tensorflow.float32

c = add(v1, v2)
