import tensorflow as tf


def gen():
    yield {"feature": tf.constant(42), "label": tf.constant(1)}


def consume(a):
    return a


dataset = tf.data.Dataset.from_generator(
    gen, output_types={"feature": tf.int32, "label": tf.int32}
)

for element in dataset:
    f = element["feature"]
    assert isinstance(f, tf.Tensor)
    assert f.dtype == tf.int32
    consume(f)
