"""Regression guard for the "layer-output flows into script-level consumer" pattern.

Companion to ``tf2_test_model_call_consume.py``, which calls ``consume(x)`` inside the model
class's ``__call__`` body. Here the consumer is at script level, after the layer call returns.
``DenseCall.getDefaultShapes``'s SSA-chain fallback recovers ``consume``'s tensor parameter
type when the PTS walk doesn't carry the synthetic ``<new>`` alloc through directly.
"""

import tensorflow as tf


def consume(t):
    pass


input_tensor = tf.constant([[1.0, 2.0, 3.0]], dtype=tf.float32)
assert input_tensor.shape == (1, 3)
assert input_tensor.dtype == tf.float32

dense = tf.keras.layers.Dense(10)
pred = dense(input_tensor)
# Runtime shape (1, 10), dtype float32 — matches the JUnit assertion.
consume(pred)
