import tensorflow as tf


def target_convert(targets):
    return targets + 1


ds = tf.data.Dataset.from_tensor_slices(tf.constant([[1, 2], [3, 4]]))
for batch in ds:
    # `batch` is the `tf.data` dataset element passed to `target_convert`.
    assert batch.shape == (2,)
    assert batch.dtype == tf.int32
    target_convert(batch)
