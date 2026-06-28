import tensorflow as tf


def target_convert(targets):
    # `targets` receives a `tf.data` dataset element at the call site.
    assert targets.shape == (2,)
    assert targets.dtype == tf.int32
    return targets + 1


ds = tf.data.Dataset.from_tensor_slices(tf.constant([[1, 2], [3, 4]]))
for batch in ds:
    target_convert(batch)
