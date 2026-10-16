import tensorflow as tf


def unpack(bbox):
    assert bbox.shape == (4,)
    assert bbox.dtype == tf.float32
    return tf.stack([bbox[0], bbox[2]])


def scale(row):
    assert row.shape == (3,)
    return row * 2


def rotate(bboxes):
    bboxes = tf.cast(bboxes, dtype=tf.float32)
    return tf.map_fn(unpack, bboxes)


rotate(tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]]))

tf.map_fn(scale, tf.constant([[1, 2, 3], [4, 5, 6]]))
scale(tf.constant([1.0, 2.0, 3.0]))


def double(row):
    assert row.shape == (1,)
    assert row.dtype == tf.float64
    return row * 2


tf.map_fn(double, tf.constant([[1.0], [2.0]], dtype=tf.float64))


def add_pair(pair):
    assert isinstance(pair, tuple)
    first, second = pair
    assert first.shape == (3,) and second.shape == (3,)
    return first + second


tf.map_fn(
    add_pair,
    (tf.constant([[1.0, 2.0, 3.0]]), tf.constant([[4.0, 5.0, 6.0]])),
    fn_output_signature=tf.float32,
)


def unused(row):
    return row * 2


stored = unused
