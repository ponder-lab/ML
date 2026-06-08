import tensorflow as tf


# Mirrors `_gather_elements_along_row` from
# `deep_recommenders/keras/models/retrieval/sbcnm.py` (the source docstring notes
# it is identical to `_take_long_axis` in `factorized_top_k`), a real-world
# recommender-systems utility, for tensor-type inference coverage.
def _gather_elements_along_row(data, column_indices):
    with tf.control_dependencies(
        [tf.assert_equal(tf.shape(data)[0], tf.shape(column_indices)[0])]
    ):
        num_row = tf.shape(data)[0]
        num_column = tf.shape(data)[1]
        num_gathered = tf.shape(column_indices)[1]
        row_indices = tf.tile(tf.expand_dims(tf.range(num_row), -1), [1, num_gathered])
        flat_data = tf.reshape(data, [-1])
        flat_indices = tf.reshape(row_indices * num_column + column_indices, [-1])
        return tf.reshape(tf.gather(flat_data, flat_indices), [num_row, num_gathered])


data = tf.constant([[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]], dtype=tf.float32)
column_indices = tf.constant([[0, 1, 2], [1, 2, 3]], dtype=tf.int32)

result = _gather_elements_along_row(data, column_indices)

assert data.shape == (2, 4) and data.dtype == tf.float32
assert column_indices.shape == (2, 3) and column_indices.dtype == tf.int32
assert result.shape == (2, 3) and result.dtype == tf.float32
