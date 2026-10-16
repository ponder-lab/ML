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
