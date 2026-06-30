import tensorflow as tf


def consume(x):
    pass


# A dataset stored in a list and read back keeps its element type when iterated (wala/ML#648): each
# sliced element of from_tensor_slices(ones([3, 4])) is (4,) float32.
ds = tf.data.Dataset.from_tensor_slices(tf.ones([3, 4]))
d = [ds, ds][0]
for elem in d:
    assert elem.shape == (4,)
    assert elem.dtype == tf.float32
    consume(elem)
