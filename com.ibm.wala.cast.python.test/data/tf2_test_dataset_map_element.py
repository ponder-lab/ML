# Test https://github.com/wala/ML/issues/164: `map` calls its function with a tuple element's
# members as separate arguments and any other element whole, and its element is what the function
# returns, so each member keeps its own type.
import tensorflow as tf


def consume_param_x(x):
    pass


def consume_param_ids(ids):
    pass


def consume_param_row(row):
    pass


def consume_param_keyed(d):
    pass


def consume_pair(p):
    pass


def consume_x(x):
    pass


def consume_ids(ids):
    pass


def consume_row(row):
    pass


def consume_keyed_a(a):
    pass


def consume_batched_x(x):
    pass


pairs = tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.zeros([8, 3], tf.int32))
)
rows = tf.data.Dataset.from_tensor_slices(tf.ones([8, 5]))
keyed = tf.data.Dataset.from_tensor_slices(
    {"a": tf.ones([8, 2]), "b": tf.zeros([8], tf.int32)}
)


def scale_pair(x, ids):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_param_x(x)
    assert ids.shape == (3,) and ids.dtype == tf.int32
    consume_param_ids(ids)
    return x * 2.0, ids


def scale_row(row):
    assert row.shape == (5,) and row.dtype == tf.float32
    consume_param_row(row)
    return row * 2.0


def pick_a(d):
    assert isinstance(d, dict)
    consume_param_keyed(d)
    return d["a"]


for p in pairs.map(scale_pair):
    assert isinstance(p, tuple) and len(p) == 2
    consume_pair(p)

for x, ids in pairs.map(scale_pair):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_x(x)
    assert ids.shape == (3,) and ids.dtype == tf.int32
    consume_ids(ids)

for row in rows.map(scale_row):
    assert row.shape == (5,) and row.dtype == tf.float32
    consume_row(row)

for a in keyed.map(pick_a):
    assert a.shape == (2,) and a.dtype == tf.float32
    consume_keyed_a(a)

for x, ids in pairs.map(scale_pair).batch(2):
    assert x.shape == (2, 4) and x.dtype == tf.float32
    consume_batched_x(x)


def consume_default_x(x):
    pass


def consume_star_last(last):
    pass


def consume_list_a(a):
    pass


def consume_mixed_first(first):
    pass


def with_default(x, ids, scale=2.0):
    assert x.shape == (4,) and x.dtype == tf.float32
    consume_default_x(x)
    return x * scale


def starred(*parts):
    last = parts[-1]
    assert last.shape == (3,) and last.dtype == tf.int32
    consume_star_last(last)
    return parts[0]


for _ in pairs.map(with_default):
    pass

for _ in pairs.map(starred):
    pass

for a, b in pairs.map(lambda x, ids: [x, ids]):
    assert a.shape == (4,) and a.dtype == tf.float32
    consume_list_a(a)

FIRST = bool(int("1"))


def first_part(*parts):
    first = parts[0]
    assert first.shape in ((4,), (5,)) and first.dtype == tf.float32
    consume_mixed_first(first)
    return first


for _ in (pairs if FIRST else rows).map(first_part):
    pass


def consume_untouched(ids):
    pass


records = tf.data.Dataset.from_tensor_slices(
    {"ids": tf.zeros([8, 3], tf.int32), "length": tf.zeros([8], tf.int32)}
)


def cast_record(e):
    e.update({"ids": tf.cast(e["ids"], tf.float32)})
    return e


for _ in records.map(cast_record):
    pass

# The function's update replaced an entry of its own dict, not of the dataset's element.
for e in records:
    assert e["ids"].shape == (3,) and e["ids"].dtype == tf.int32
    consume_untouched(e["ids"])


def consume_fifth(e):
    pass


fives = tf.data.Dataset.from_tensor_slices(
    (
        tf.ones([8, 2]),
        tf.ones([8, 2]),
        tf.ones([8, 2]),
        tf.ones([8, 2]),
        tf.ones([8, 6]),
    )
)


def fifth_or_default(a, b, c, d, e=tf.zeros([7], tf.int32)):
    return e


# A tuple element longer than the analysis spreads is passed whole to no call, so the mapped
# element is unknown rather than the default.
for e in fives.map(fifth_or_default):
    assert e.shape == (6,) and e.dtype == tf.float32
    consume_fifth(e)


def consume_raw_second(t):
    pass


def consume_raw_batched_second(t):
    pass


def consume_raw_first(t):
    pass


def pair_with_raw(a, b):
    return a, tf.raw_ops.Abs(x=b)


# A member made by an operation the analysis does not model is unknown, not absent.
raws = tf.data.Dataset.from_tensor_slices((tf.ones([8, 4]), tf.ones([8, 3]))).map(
    pair_with_raw
)
for e in raws:
    assert e[1].shape == (3,) and e[1].dtype == tf.float32
    consume_raw_second(e[1])
    assert e[0].shape == (4,) and e[0].dtype == tf.float32
    consume_raw_first(e[0])
for e in raws.batch(2):
    assert e[1].shape == (2, 3) and e[1].dtype == tf.float32
    consume_raw_batched_second(e[1])


def consume_raw_slice_second(t):
    pass


# The same for a member of a structure the dataset is built from.
for e in tf.data.Dataset.from_tensor_slices(
    (tf.ones([8, 4]), tf.raw_ops.Abs(x=tf.ones([8, 3])))
):
    assert e[1].shape == (3,) and e[1].dtype == tf.float32
    consume_raw_slice_second(e[1])


def consume_none_aux(t):
    pass


def consume_none_mapped(t):
    pass


# A None member is no tensor, whether the dataset is built with it or a function returns it.
for e in tf.data.Dataset.from_tensor_slices({"x": tf.ones([4, 2]), "aux": None}):
    assert e["aux"] is None
    consume_none_aux(e["aux"])
for e in (
    tf.data.Dataset.from_tensor_slices(tf.ones([4, 2]))
    .map(lambda x: (x, None))
    .batch(2)
):
    assert e[1] is None
    consume_none_mapped(e[1])


def consume_raw_int_key(t):
    pass


def int_keyed_with_raw(a, b):
    return {0: a, 1: tf.raw_ops.Abs(x=b)}


# The same for a member of a dict keyed by integers.
for e in tf.data.Dataset.from_tensor_slices((tf.ones([8, 4]), tf.ones([8, 3]))).map(
    int_keyed_with_raw
):
    assert e[1].shape == (3,) and e[1].dtype == tf.float32
    consume_raw_int_key(e[1])
