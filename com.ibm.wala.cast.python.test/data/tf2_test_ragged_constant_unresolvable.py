import json

import tensorflow as tf


def consume(rt):
    pass


# `pylist` comes from an opaque `json.loads`, so neither the ragged structure nor the dtype is
# resolvable to the analysis. `RaggedConstant` floors the shape to ⊤ and the dtype to ⊤ rather than
# aborting with "Expected a list or tuple" / "Empty points-to set". wala/ML#612.
pylist = json.loads("[[1, 2], [3]]")
rt = tf.ragged.constant(pylist)

# At runtime the ragged tensor is precise; the static analysis floors both axes to ⊤ because the
# `pylist` is opaque (the captured gap this fixture guards).
assert rt.shape.as_list() == [2, None]
assert rt.dtype == tf.int32

consume(rt)
