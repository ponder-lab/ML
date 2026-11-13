# `rejection_resample` calls its `class_func` on the receiver's element, the method and
# `tf.data.experimental.rejection_resample` handed to `apply` alike, and yields `(class, element)`
# pairs (wala/ML#1214).
import tensorflow as tf


def consume_class_arg(v):
    pass


def consume_applied_class_arg(v):
    pass


def consume_class(c):
    pass


def consume_element(e):
    pass


def consume_applied_class(c):
    pass


def consume_applied_element(e):
    pass


def class_of(v):
    assert v.shape == (4,) and v.dtype == tf.float32, v
    consume_class_arg(v)
    return tf.constant(0, tf.int32)


def applied_class_of(v):
    assert v.shape == (4,) and v.dtype == tf.float32, v
    consume_applied_class_arg(v)
    return tf.constant(0, tf.int32)


flat = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))

for c, e in flat.rejection_resample(
    class_func=class_of, target_dist=[1.0, 0.0], initial_dist=[1.0, 0.0], seed=0
):
    assert c.shape == () and c.dtype == tf.int32, c
    assert e.shape == (4,) and e.dtype == tf.float32, e
    consume_class(c)
    consume_element(e)

for c, e in flat.apply(
    tf.data.experimental.rejection_resample(
        class_func=applied_class_of,
        target_dist=[1.0, 0.0],
        initial_dist=[1.0, 0.0],
        seed=0,
    )
):
    assert c.shape == () and c.dtype == tf.int32, c
    assert e.shape == (4,) and e.dtype == tf.float32, e
    consume_applied_class(c)
    consume_applied_element(e)
