import tensorflow as tf


def gen():
    ragged_tensor = tf.ragged.constant([[1, 2], [3]])
    yield 42, ragged_tensor


def add(a, b):
    return a + b


dataset = tf.data.Dataset.from_generator(
    gen,
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.int32),
        tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32),
    ),
)

for element in dataset:
    assert type(element) == tuple
    assert element[0].shape == ()
    assert element[0].dtype == tf.int32
    assert element[1].shape.as_list() == [2, None]
    assert element[1].dtype == tf.int32
    c = add(element, element)
