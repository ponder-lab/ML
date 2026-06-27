import json

import tensorflow as tf


def consume(rt):
    pass


# Both `flat_values` and `nested_value_rowids` come from opaque `json.loads`, so neither the ragged
# structure nor the dtype is resolvable. `RaggedFromNestedValueRowIds` floors the shape to ⊤ and the
# dtype to ⊤ rather than aborting with "Could not calculate shapes" / "Could not determine dtypes".
# wala/ML#612.
flat_values = json.loads('["a", "b", "c", "d", "e", "f", "g"]')
nested_value_rowids = json.loads("[[0, 0, 1, 2], [0, 1, 1, 2, 2, 2, 3]]")
rt = tf.RaggedTensor.from_nested_value_rowids(flat_values, nested_value_rowids)
consume(rt)
