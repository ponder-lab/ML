import tensorflow as tf


def add(a, b):
    return a + b


dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3], [4, 5, 6])).shuffle(3).batch(2)

for element_a, element_b in dataset:
    assert isinstance(element_a, tf.Tensor)
    assert isinstance(element_b, tf.Tensor)
    assert element_a.shape == (2,) or element_a.shape == (1,)
    assert element_b.shape == (2,) or element_b.shape == (1,)
    assert element_a.dtype == tf.int32
    assert element_b.dtype == tf.int32
    c = add(element_a, element_b)
