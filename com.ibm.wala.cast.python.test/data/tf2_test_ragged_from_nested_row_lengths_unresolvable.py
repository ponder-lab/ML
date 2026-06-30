import json

import tensorflow as tf


def consume(a):
    pass


# `nested_row_lengths` comes from an opaque `json.loads`, so the ragged structure is unresolvable to
# the analysis. `RaggedFromNested` floors the shape to ⊤ (unknown) rather than aborting the whole
# analysis with "Could not calculate shapes". wala/ML#612.
x = [10, 20, 30, 40, 50, 60, 70]
nested_row_lengths = json.loads("[[2, 1, 0, 2], [2, 0, 3, 1, 1]]")
arg = tf.RaggedTensor.from_nested_row_lengths(x, nested_row_lengths)

# At runtime the ragged tensor is precise; the static analysis floors the shape to ⊤ because
# `nested_row_lengths` is opaque (the captured gap this fixture guards). The dtype still resolves to
# `int32` from the resolvable `x`.
assert arg.shape.as_list() == [4, None, None]
assert arg.dtype == tf.int32

consume(arg)
