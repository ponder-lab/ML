import tensorflow as tf


def consume(x):
    pass


# wala/ML#545 regression guard for the `DynamicDim` fallback branch in
# `RaggedFromRowLengths`'s nrows inference. See `tf2_test_ragged_from_row_starts_dynamic.py`
# for the shared rationale.
try:
    row_lengths = tf.keras.Input(shape=(8,), dtype=tf.int64)
    values = tf.constant([3, 1, 4, 1, 5, 9, 2, 6])
    r = tf.RaggedTensor.from_row_lengths(values, row_lengths)
    consume(r)
except (TypeError, ValueError):
    # Intentional swallow: the symbolic input above only exercises the analyzer; the
    # runtime rejects it. No state to recover.
    pass
