import json
import os
import tempfile

import tensorflow as tf


def consume(rt):
    pass


# `pylist` is read from a file at runtime, so its contents are not statically knowable: a genuinely
# content-dependent (opaque) source, in the sense of wala/ML#370. The unmodeled file read leaves its
# points-to set empty, so `RaggedConstant` floors both the shape and the dtype to ⊤ rather than
# aborting with "Empty points-to set". wala/ML#612.
fd, path = tempfile.mkstemp(suffix=".json")
with os.fdopen(fd, "w") as fh:
    fh.write("[[1, 2], [3]]")
with open(path) as fh:
    pylist = json.loads(fh.read())
os.remove(path)

rt = tf.ragged.constant(pylist)

# At runtime the ragged tensor is precise; the static analysis floors both axes to ⊤ because the
# file-sourced `pylist` is opaque (the captured gap this fixture guards).
assert rt.shape.as_list() == [2, None]
assert rt.dtype == tf.int32

consume(rt)
