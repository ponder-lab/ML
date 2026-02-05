import tensorflow


def add(a, b):
    assert a.shape.as_list() == [1, 2]
    assert b.shape.as_list() == [2, 2]
    return a + b


c = add(tensorflow.ones([1, 2]), tensorflow.ones([2, 2]))  #  [[2., 2.], [2., 2.]]
assert c.shape.as_list() == [2, 2]
assert c.dtype == tensorflow.float32
