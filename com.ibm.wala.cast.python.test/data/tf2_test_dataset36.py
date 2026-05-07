import tensorflow as tf


class C:

    def __init__(self, some_iter):
        self.some_iter = some_iter

    def __str__(self):
        return str(self.some_iter)


def id1(a):
    return a


def id2(a):
    return a


def gen():
    yield "42", tf.constant("43")


dataset = tf.data.Dataset.from_generator(gen, output_types=(tf.string, tf.string))

my_iter = iter(dataset)
c = C(my_iter)
length = 1

for _ in range(length):
    x, y = next(c.some_iter)
    assert isinstance(x, tf.Tensor)
    assert x.shape == ()
    assert x.dtype == tf.string
    id1(x)
    assert isinstance(y, tf.Tensor)
    assert y.shape == ()
    assert y.dtype == tf.string
    id2(y)
