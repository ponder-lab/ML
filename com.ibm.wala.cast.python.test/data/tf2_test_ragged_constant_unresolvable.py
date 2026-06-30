import json

import tensorflow as tf


def consume(rt):
    pass


# `json.loads` is not modeled, so `pylist`'s points-to set is empty even though the values are
# inline. `RaggedConstant` floors the shape and the dtype to ⊤ rather than aborting with "Empty
# points-to set". wala/ML#612.
pylist = json.loads("[[1, 2], [3]]")
rt = tf.ragged.constant(pylist)

# At runtime the ragged tensor is precise; the static analysis floors both axes to ⊤ because
# `json.loads` is unmodeled (the captured gap this fixture guards).
assert rt.shape.as_list() == [2, None]
assert rt.dtype == tf.int32

consume(rt)
