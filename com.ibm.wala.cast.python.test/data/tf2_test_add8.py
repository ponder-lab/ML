import tensorflow


def add(a, b):
    return a + b


x = tensorflow.ones([1, 2])
assert x.shape.as_list() == [1, 2]
y = tensorflow.ones([2, 2])
assert y.shape.as_list() == [2, 2]
c = add(x, y)  #  [[2., 2.], [2., 2.]]
