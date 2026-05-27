import tensorflow as tf


def consume(x):
    pass


# wala/ML#545 regression guard: when `row_starts` has a non-`NumericDim`
# first dim (here `DynamicDim` from a Keras Input's batch axis), the
# `RaggedFromRowStarts` generator must emit `DynamicDim.INSTANCE` for the
# inferred nrows instead of raw `null`.
#
# At runtime, `tf.RaggedTensor.from_row_starts` rejects a symbolic
# `KerasTensor` arg — the `try/except` lets the file run to completion
# per `CLAUDE.md`'s fixture guideline while still letting the static
# analyzer walk the `from_row_starts` call below.
try:
    row_starts = tf.keras.Input(shape=(8,), dtype=tf.int64)
    values = tf.constant([3, 1, 4, 1, 5, 9, 2, 6])
    r = tf.RaggedTensor.from_row_starts(values, row_starts)
    consume(r)
except (TypeError, ValueError):
    # Intentional swallow: the symbolic input above only exercises the analyzer; the
    # runtime rejects it. No state to recover.
    pass
