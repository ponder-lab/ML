import tensorflow as tf


def f(a):
    pass


# Regression fixture for the wala/ML#518 throw path at
# `RaggedFromNestedValueRowIds:123` (nested_nrows arg collection). The fixture
# passes a non-numeric string in `nested_nrows`, which TF would reject at
# runtime but which the static analyzer's `Long.parseLong((String) val)` must
# handle: the post-#518 contract is to catch the `NumberFormatException` and
# rethrow as `IllegalStateException` (with the original NFE as `cause`).
#
# The Python `tf.RaggedTensor.from_nested_value_rowids` call would fail at
# runtime; the analyzer doesn't execute it, just walks the AST and resolves
# constants through the PA. The non-numeric "abc" reaches the throw site.
rt = tf.RaggedTensor.from_nested_value_rowids(
    flat_values=[1, 2, 3, 4],
    nested_value_rowids=[[0, 0, 1, 1]],
    nested_nrows=["abc"],
)
f(rt)
