# A `padded_shapes` the program supplies to `tf.data.experimental.bucket_by_sequence_length` handed
# to `apply` declares the padded shape, as it does for the method (wala/ML#1214). It is alone here,
# since two such transformations in one program share the summary's dataset.
import tensorflow as tf


def consume_bucketed_applied_padded(x):
    pass


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))

# A declared padded shape pads each element to it.
for x in flat.apply(
    tf.data.experimental.bucket_by_sequence_length(
        element_length_func=lambda v: tf.shape(v)[0],
        bucket_boundaries=[3],
        bucket_batch_sizes=[2, 2],
        padded_shapes=[5],
    )
):
    assert x.shape == (2, 5) and x.dtype == tf.float32, x
    consume_bucketed_applied_padded(x)
