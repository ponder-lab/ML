# A dataset whose element is fed back into a dataset built from it: a `batch` applied in a loop to
# its own result, and an element passed to `from_tensor_slices` again. The analysis copies each
# element's structure, which must still be finite.
import tensorflow as tf


def consume_rebatched(x):
    pass


def consume_resliced(x):
    pass


rebatched = tf.data.Dataset.from_tensor_slices((tf.ones((8, 4)), tf.zeros((8,))))
for _ in range(3):
    rebatched = rebatched.batch(2)
for x, y in rebatched:
    assert x.shape == (2, 2, 2, 4) and x.dtype == tf.float32
    consume_rebatched(x)

resliced = tf.data.Dataset.from_tensor_slices(tf.ones((4, 4)))
for _ in range(1):
    element = next(iter(resliced))
    resliced = tf.data.Dataset.from_tensor_slices(element)
for value in resliced:
    assert value.shape == () and value.dtype == tf.float32
    consume_resliced(value)


# A structure whose points-to set holds itself three ways, built in a loop: the copy of the
# element's structure is bounded.
def consume_self_referencing(x):
    pass


config = {"x": tf.ones((2, 3))}
for _ in range(2):
    config = {"a": config, "b": config, "c": config}
for record in tf.data.Dataset.from_tensor_slices(config):
    leaf = record["a"]["b"]["x"]
    assert leaf.shape == (3,) and leaf.dtype == tf.float32
    consume_self_referencing(leaf)
