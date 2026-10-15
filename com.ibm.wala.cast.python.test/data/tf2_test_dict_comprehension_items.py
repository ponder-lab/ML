# Test for https://github.com/wala/ML/issues/1071.
import tensorflow as tf


def sink(x):
    assert x.shape == (1, 3, 1)
    assert x.dtype == tf.float32
    return x


def losses(batch):
    return sink(tf.expand_dims(batch["audios"], -1))


records = tf.data.Dataset.from_tensor_slices(
    {"audios": tf.ones((4, 3), dtype=tf.float32), "ids": tf.zeros((4,), dtype=tf.int32)}
).batch(2)

for batch in records:
    for i in tf.range(2):
        reduced_batch = {k: v[i : i + 1] for k, v in batch.items()}
        losses(reduced_batch)
