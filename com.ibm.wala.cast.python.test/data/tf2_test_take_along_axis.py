# Minimal fixture mirroring `_take_long_axis` from
# `LongmaoTeamTf/deep_recommenders/keras/models/retrieval/factorized_top_k.py`.
# Tests caller-side type propagation when the function takes two tensor params
# (`arr` and `indices`) with different dtypes (float32 and int32).
import tensorflow as tf


def _take_long_axis(arr, indices):
    """Take elements from arr at indices specified by indices (2D)."""
    row_indices = tf.tile(
        tf.expand_dims(tf.range(tf.shape(indices)[0]), 1), [1, tf.shape(indices)[1]]
    )
    gather_indices = tf.concat(
        [tf.reshape(row_indices, (-1, 1)), tf.reshape(indices, (-1, 1))], axis=1
    )
    return tf.reshape(tf.gather_nd(arr, gather_indices), tf.shape(indices))


# Driver: build known input tensors and pass them through.
arr = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=tf.float32)
indices = tf.constant([[0, 1], [2, 0]], dtype=tf.int32)

assert arr.shape == (2, 3), f"arr shape was {arr.shape}"
assert arr.dtype == tf.float32
assert indices.shape == (2, 2), f"indices shape was {indices.shape}"
assert indices.dtype == tf.int32

result = _take_long_axis(arr, indices)
assert result.shape == (2, 2)
assert result.dtype == tf.float32
