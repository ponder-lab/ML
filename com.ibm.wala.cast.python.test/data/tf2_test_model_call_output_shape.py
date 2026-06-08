import tensorflow as tf


def consume_positional(x):
    pass


def consume_keyword(x):
    pass


# A functional `tf.keras.Model` whose output shape differs from its input shape:
# `Dense(3)` maps a `(*, 2)` input to a `(*, 3)` output. Exercises `ModelCall`
# recovering the model's output generator (and thus the transformed shape) rather
# than passing the call's input shape through. See wala/ML#537.
def positional():
    input_node = tf.keras.Input(shape=(2,), dtype=tf.float32)
    output_node = tf.keras.layers.Dense(3)(input_node)
    model = tf.keras.Model(input_node, output_node)

    out = model(tf.ones((1, 2)))
    assert out.shape.as_list() == [1, 3]
    assert out.dtype == tf.float32
    consume_positional(out)


def keyword():
    input_node = tf.keras.Input(shape=(2,), dtype=tf.float32)
    output_node = tf.keras.layers.Dense(3)(input_node)
    model = tf.keras.Model(outputs=output_node, inputs=input_node)

    out = model(tf.ones((1, 2)))
    assert out.shape.as_list() == [1, 3]
    assert out.dtype == tf.float32
    consume_keyword(out)


if __name__ == "__main__":
    positional()
    keyword()
