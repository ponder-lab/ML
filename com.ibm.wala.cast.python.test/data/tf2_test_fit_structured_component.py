# `fit` on a batched `tf.data.Dataset` built by `from_tensor_slices`, with Keras's default
# `train_step`, whose inputs component is itself a structure: a tuple, `((x, mask), y)`, or a dict,
# `({"ids": ids, "m": m}, y)`. The model's `call` then receives that structure, a tuple or a dict of
# tensors, and not a tensor; its members are tensors. With flat inputs, `(x, y)`, `call` receives a
# tensor.
import tensorflow as tf


def loss(y_true, y_pred):
    return tf.reduce_mean(y_pred)


def consume_pair_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2
    assert not isinstance(inputs, tf.Tensor)


def consume_pair_x(x):
    # Traced by `fit` with the batch axis unspecified.
    assert x.shape[1:] == (4,) and x.dtype == tf.float32


def consume_pair_mask(mask):
    assert mask.shape[1:] == (3,) and mask.dtype == tf.float32


def consume_pair_first(first):
    assert first.shape[1:] == (4,) and first.dtype == tf.float32


class Pair(tf.keras.Model):
    def call(self, inputs):
        consume_pair_inputs(inputs)
        consume_pair_first(inputs[0])
        x, mask = inputs
        consume_pair_x(x)
        consume_pair_mask(mask)
        return x


pair = Pair()
pair.compile(optimizer="sgd", loss=loss)
pair.fit(
    tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
    ).batch(2),
    epochs=1,
    verbose=0,
)


def consume_mixed_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_mixed_ids(ids):
    assert ids.shape[1:] == (3,) and ids.dtype == tf.int32


class Mixed(tf.keras.Model):
    def call(self, inputs):
        consume_mixed_inputs(inputs)
        x, ids = inputs
        consume_mixed_ids(ids)
        return x


mixed = Mixed()
mixed.compile(optimizer="sgd", loss=loss)
mixed.fit(
    tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3], tf.int32)), tf.zeros([8]))
    ).batch(2),
    epochs=1,
    verbose=0,
)


def consume_keyed_inputs(inputs):
    assert isinstance(inputs, dict) and sorted(inputs) == ["ids", "m"]


def consume_keyed_ids(ids):
    assert ids.shape[1:] == (3,) and ids.dtype == tf.int32


def consume_keyed_m(m):
    assert m.shape[1:] == (5,) and m.dtype == tf.float32


class Keyed(tf.keras.Model):
    def call(self, inputs):
        consume_keyed_inputs(inputs)
        consume_keyed_ids(inputs["ids"])
        consume_keyed_m(inputs["m"])
        return inputs["m"]


keyed = Keyed()
keyed.compile(optimizer="sgd", loss=loss)
keyed.fit(
    tf.data.Dataset.from_tensor_slices(
        ({"ids": tf.ones([8, 3], tf.int32), "m": tf.ones([8, 5])}, tf.zeros([8]))
    ).batch(2),
    epochs=1,
    verbose=0,
)


def consume_flat_inputs(inputs):
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


class Flat(tf.keras.Model):
    def call(self, inputs):
        consume_flat_inputs(inputs)
        return inputs


flat = Flat()
flat.compile(optimizer="sgd", loss=loss)
flat.fit(
    tf.data.Dataset.from_tensor_slices((tf.ones([8, 4]), tf.zeros([8]))).batch(2),
    epochs=1,
    verbose=0,
)


# A structure the element's component takes through `map`.
def consume_map_pair_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_map_pair_b(b):
    assert b.shape[1:] == (3,) and b.dtype == tf.float32


class MapPair(tf.keras.Model):
    def call(self, inputs):
        consume_map_pair_inputs(inputs)
        a, b = inputs
        consume_map_pair_b(b)
        return a


triples = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.ones([8, 3]), tf.zeros([8]))
)
map_pair = MapPair()
map_pair.compile(optimizer="sgd", loss=loss)
map_pair.fit(triples.map(lambda a, b, y: ((a, b), y)).batch(2), epochs=1, verbose=0)


def consume_map_flat_inputs(inputs):
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


class MapFlat(tf.keras.Model):
    def call(self, inputs):
        consume_map_flat_inputs(inputs)
        return inputs


map_flat = MapFlat()
map_flat.compile(optimizer="sgd", loss=loss)
map_flat.fit(triples.map(lambda a, b, y: (a, y)).batch(2), epochs=1, verbose=0)


# A structure the element's component takes through `zip`.
def consume_zip_pair_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_zip_pair_m(m):
    assert m.shape[1:] == (3,) and m.dtype == tf.float32


class ZipPair(tf.keras.Model):
    def call(self, inputs):
        consume_zip_pair_inputs(inputs)
        x, m = inputs
        consume_zip_pair_m(m)
        return x


zip_x = tf.data.Dataset.from_tensor_slices(tf.ones([8, 4]))
zip_m = tf.data.Dataset.from_tensor_slices(tf.ones([8, 3]))
zip_y = tf.data.Dataset.from_tensor_slices(tf.zeros([8]))
zip_pair = ZipPair()
zip_pair.compile(optimizer="sgd", loss=loss)
zip_pair.fit(tf.data.Dataset.zip(((zip_x, zip_m), zip_y)).batch(2), epochs=1, verbose=0)


def consume_zip_flat_inputs(inputs):
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


class ZipFlat(tf.keras.Model):
    def call(self, inputs):
        consume_zip_flat_inputs(inputs)
        return inputs


zip_flat = ZipFlat()
zip_flat.compile(optimizer="sgd", loss=loss)
zip_flat.fit(tf.data.Dataset.zip((zip_x, zip_y)).batch(2), epochs=1, verbose=0)


def consume_zip_element_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_zip_element_m(m):
    assert m.shape[1:] == (3,) and m.dtype == tf.float32


class ZipElement(tf.keras.Model):
    def call(self, inputs):
        consume_zip_element_inputs(inputs)
        x, m = inputs
        consume_zip_element_m(m)
        return x


# The first zipped dataset's whole element is the tuple component.
zip_element = ZipElement()
zip_element.compile(optimizer="sgd", loss=loss)
zip_element.fit(
    tf.data.Dataset.zip(
        (tf.data.Dataset.from_tensor_slices((tf.ones([8, 4]), tf.ones([8, 3]))), zip_y)
    ).batch(2),
    epochs=1,
    verbose=0,
)


def consume_zip_deep_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_zip_deep_pair(pair):
    assert isinstance(pair, tuple) and len(pair) == 2


def consume_zip_deep_z(z):
    assert z.shape[1:] == (5,) and z.dtype == tf.float32


class ZipDeep(tf.keras.Model):
    def call(self, inputs):
        consume_zip_deep_inputs(inputs)
        pair, z = inputs
        consume_zip_deep_pair(pair)
        consume_zip_deep_z(z)
        return z


# The first zipped dataset's element is `((x, m), z)`, so the component's first member is a tuple.
zip_deep = ZipDeep()
zip_deep.compile(optimizer="sgd", loss=loss)
zip_deep.fit(
    tf.data.Dataset.zip(
        (
            tf.data.Dataset.from_tensor_slices(
                ((tf.ones([8, 4]), tf.ones([8, 3])), tf.ones([8, 5]))
            ),
            zip_y,
        )
    ).batch(2),
    epochs=1,
    verbose=0,
)


# A dataset chosen by a branch.
def pair_dataset():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 3])), tf.zeros([8]))
    ).batch(2)


def other_pair_dataset():
    return tf.data.Dataset.from_tensor_slices(
        ((tf.ones([8, 4]), tf.ones([8, 5])), tf.zeros([8]))
    ).batch(2)


def flat_dataset():
    return tf.data.Dataset.from_tensor_slices((tf.ones([8, 6]), tf.zeros([8]))).batch(2)


def consume_branch_pair_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


class BranchPair(tf.keras.Model):
    def call(self, inputs):
        consume_branch_pair_inputs(inputs)
        x, m = inputs
        return x


def fit_branch_pair(flag):
    model = BranchPair()
    model.compile(optimizer="sgd", loss=loss)
    ds = pair_dataset() if flag else other_pair_dataset()
    model.fit(ds, epochs=1, verbose=0)


# One call site, so one frame where the choice is either dataset.
for flag in (True, False):
    fit_branch_pair(flag)


def consume_branch_mixed_inputs(inputs):
    # A tuple when the structured dataset is chosen and a tensor when the flat one is.
    assert isinstance(inputs, (tuple, tf.Tensor))


class BranchMixed(tf.keras.Model):
    def call(self, inputs):
        consume_branch_mixed_inputs(inputs)
        return inputs[0]


def fit_branch_mixed(flag):
    model = BranchMixed()
    model.compile(optimizer="sgd", loss=loss)
    ds = pair_dataset() if flag else flat_dataset()
    model.fit(ds, epochs=1, verbose=0)


# One call site, so one frame where the choice is either dataset.
for flag in (True, False):
    fit_branch_mixed(flag)


def consume_split_mixed_inputs(inputs):
    # A tuple when the structured dataset is chosen and a tensor when the flat one is.
    assert isinstance(inputs, (tuple, tf.Tensor))


class SplitMixed(tf.keras.Model):
    def call(self, inputs):
        consume_split_mixed_inputs(inputs)
        return inputs[0]


def fit_split_mixed(flag):
    model = SplitMixed()
    model.compile(optimizer="sgd", loss=loss)
    ds = pair_dataset() if flag else flat_dataset()
    model.fit(ds, epochs=1, verbose=0)


# Two call sites, so a frame per choice.
fit_split_mixed(True)
fit_split_mixed(False)


# An iterator over the dataset.
def consume_iter_pair_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


class IterPair(tf.keras.Model):
    def call(self, inputs):
        consume_iter_pair_inputs(inputs)
        x, m = inputs
        return x


iter_pair = IterPair()
iter_pair.compile(optimizer="sgd", loss=loss)
iter_pair.fit(iter(pair_dataset().repeat()), steps_per_epoch=2, epochs=1, verbose=0)


# `from_generator` with an `output_signature` and with the legacy `output_types`.
def pair_batches():
    yield (tf.ones([2, 4]), tf.ones([2, 3])), tf.zeros([2])


def flat_batches():
    yield tf.ones([2, 4]), tf.zeros([2])


def consume_signature_pair_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


def consume_signature_pair_m(m):
    assert m.shape[1:] == (3,) and m.dtype == tf.float32


class SignaturePair(tf.keras.Model):
    def call(self, inputs):
        consume_signature_pair_inputs(inputs)
        x, m = inputs
        consume_signature_pair_m(m)
        return x


signature_pair = SignaturePair()
signature_pair.compile(optimizer="sgd", loss=loss)
signature_pair.fit(
    tf.data.Dataset.from_generator(
        pair_batches,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                tf.TensorSpec(shape=(None, 3), dtype=tf.float32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    ),
    epochs=1,
    verbose=0,
)


def consume_signature_flat_inputs(inputs):
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


class SignatureFlat(tf.keras.Model):
    def call(self, inputs):
        consume_signature_flat_inputs(inputs)
        return inputs


signature_flat = SignatureFlat()
signature_flat.compile(optimizer="sgd", loss=loss)
signature_flat.fit(
    tf.data.Dataset.from_generator(
        flat_batches,
        output_signature=(
            tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        ),
    ),
    epochs=1,
    verbose=0,
)


def consume_types_pair_inputs(inputs):
    assert isinstance(inputs, tuple) and len(inputs) == 2


class TypesPair(tf.keras.Model):
    def call(self, inputs):
        consume_types_pair_inputs(inputs)
        x, m = inputs
        return x


types_pair = TypesPair()
types_pair.compile(optimizer="sgd", loss=loss)
types_pair.fit(
    tf.data.Dataset.from_generator(
        pair_batches,
        output_types=((tf.float32, tf.float32), tf.float32),
        output_shapes=(
            (tf.TensorShape([None, 4]), tf.TensorShape([None, 3])),
            tf.TensorShape([None]),
        ),
    ),
    epochs=1,
    verbose=0,
)


def consume_types_flat_inputs(inputs):
    assert isinstance(inputs, tf.Tensor)
    assert inputs.shape[1:] == (4,) and inputs.dtype == tf.float32


class TypesFlat(tf.keras.Model):
    def call(self, inputs):
        consume_types_flat_inputs(inputs)
        return inputs


types_flat = TypesFlat()
types_flat.compile(optimizer="sgd", loss=loss)
types_flat.fit(
    tf.data.Dataset.from_generator(
        flat_batches,
        output_types=(tf.float32, tf.float32),
        output_shapes=(tf.TensorShape([None, 4]), tf.TensorShape([None])),
    ),
    epochs=1,
    verbose=0,
)
