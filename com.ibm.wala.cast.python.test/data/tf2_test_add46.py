import tensorflow
from tensorflow.python.framework.ops import Tensor


def value_index(a, b):
    return a.value_index + b.value_index


# From https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/Graph#using_graphs_directly_deprecated
g = tensorflow.Graph()
with g.as_default():
    # Defines operation and tensor in graph
    c = tensorflow.constant(30.0)
    assert c.graph is g

arg1 = Tensor(g.get_operations()[0], 0, tensorflow.float32)
assert isinstance(arg1, Tensor)
assert arg1.shape == ()
assert arg1.dtype == tensorflow.float32

arg2 = Tensor(g.get_operations()[0], 0, tensorflow.float32)
assert isinstance(arg2, Tensor)
assert arg2.shape == ()
assert arg2.dtype == tensorflow.float32

result = value_index(arg1, arg2)
