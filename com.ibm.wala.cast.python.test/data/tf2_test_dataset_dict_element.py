# Test for wala/ML#993 (remainder): the element of a dataset whose elements are dicts, or tuples
# of dicts, read in a loop over the dataset. Each variant sinks the element's `ids`.
import tensorflow as tf


def consume_dict(x):
    pass


def consume_dict_batched(x):
    pass


def consume_tuple_tensor(x):
    pass


def consume_tuple_dict(x):
    pass


def consume_tuple_dict_batched(x):
    pass


def consume_spec_step(x):
    pass


def consume_whole_step(x):
    pass


ids = tf.constant([[1, 2, 3], [4, 5, 6]])
length = tf.constant([3, 3])

# A dataset of dicts.
for element in tf.data.Dataset.from_tensor_slices({"ids": ids, "length": length}):
    consume_dict(element["ids"])
    assert element["ids"].shape == (3,) and element["ids"].dtype == tf.int32

for element in tf.data.Dataset.from_tensor_slices({"ids": ids, "length": length}).batch(
    2
):
    consume_dict_batched(element["ids"])
    assert element["ids"].shape == (2, 3)

# A dataset of tuples of tensors.
for a, b in tf.data.Dataset.from_tensor_slices((ids, length)):
    consume_tuple_tensor(a)
    assert a.shape == (3,)

# A dataset of tuples of dicts.
for source, target in tf.data.Dataset.from_tensor_slices(
    ({"ids": ids, "length": length}, {"ids": ids, "length": length})
):
    consume_tuple_dict(source["ids"])
    assert source["ids"].shape == (3,)

batched = tf.data.Dataset.from_tensor_slices(
    ({"ids": ids, "length": length}, {"ids": ids, "length": length})
).batch(2)
for source, target in batched:
    consume_tuple_dict_batched(source["ids"])
    assert source["ids"].shape == (2, 3)


# The element_spec hop: a step whose signature is the dataset's element_spec, called in the loop.
@tf.function(input_signature=batched.element_spec)
def step(source, target):
    consume_spec_step(source["ids"])
    tf.debugging.assert_equal(tf.shape(source["ids"]), [2, 3])
    return source["ids"]


for source, target in batched:
    step(source, target)


# A whole dict element passed into a function and subscripted there.
def whole_step(element):
    consume_whole_step(element["ids"])
    assert element["ids"].shape == (2, 3) and element["ids"].dtype == tf.int32
    return element["ids"]


for element in tf.data.Dataset.from_tensor_slices({"ids": ids, "length": length}).batch(
    2
):
    whole_step(element)


def consume_shuffled(x):
    pass


def consume_mapped(x):
    pass


def consume_two_datasets(x):
    pass


def consume_keyword_step(x):
    pass


# A dict dataset through an operation that keeps its element structure: the component resolves
# through the operation to its source.
for element in tf.data.Dataset.from_tensor_slices(
    {"ids": ids, "length": length}
).shuffle(2):
    consume_shuffled(element["ids"])
    assert element["ids"].shape == (3,)

# A dict dataset through a `map` whose function returns its element: the component resolves
# through the map to its source.
for element in tf.data.Dataset.from_tensor_slices({"ids": ids, "length": length}).map(
    lambda e: e
):
    consume_mapped(element["ids"])
    assert element["ids"].shape == (3,) and element["ids"].dtype == tf.int32


# A function fed dict elements of two different datasets by two callers: each call is its own
# context and resolves its own dataset's element.
def two_datasets_step(element):
    consume_two_datasets(element["ids"])
    assert element["ids"].shape in ((3,), (2,)) and element["ids"].dtype == tf.int32


other = tf.data.Dataset.from_tensor_slices(
    {"ids": tf.constant([[7, 8]]), "length": length[:1]}
)
for element in tf.data.Dataset.from_tensor_slices({"ids": ids, "length": length}):
    two_datasets_step(element)
for element in other:
    two_datasets_step(element)


# An element supplied to its parameter by keyword is typed as well.
def keyword_step(element):
    consume_keyword_step(element["ids"])
    assert element["ids"].shape == (3,) and element["ids"].dtype == tf.int32


for element in tf.data.Dataset.from_tensor_slices({"ids": ids, "length": length}):
    keyword_step(element=element)
