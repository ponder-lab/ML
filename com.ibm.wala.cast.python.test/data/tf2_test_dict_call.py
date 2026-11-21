# Test https://github.com/wala/ML/issues/1307.
import tensorflow as tf


def by_keywords(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_keywords_other(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def by_display(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_pairs(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_pairs_other(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def by_named_pairs(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def by_tuple_of_pairs(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_copy(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_copy_other(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def by_unpacking(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_copy_and_keyword(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_copy_and_keyword_other(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def by_keywords_get(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_first_call(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_second_call(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def by_function(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_empty_then_store(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_parameter_key(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_items(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_alias(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_nested_call(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_copy_of_call(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_copy_of_pairs(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_keyword_display(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_constant_value(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_two_unpackings(x):
    assert x.shape == (2, 4) and x.dtype == tf.float32


def by_two_unpackings_other(x):
    assert x.shape == (2,) and x.dtype == tf.int32


def merge(first, second):
    return dict(**first, **second)


def build(first, second):
    return dict(boxes=first, labels=second)


def pair(key, value):
    return dict([(key, value)])


boxes = tf.ones([2, 4])
labels = tf.zeros([2], tf.int32)

by_keywords(dict(boxes=boxes, labels=labels)["boxes"])
by_keywords_other(dict(boxes=boxes, labels=labels)["labels"])
by_display({"boxes": boxes, "labels": labels}["boxes"])
by_pairs(dict([("boxes", boxes), ("labels", labels)])["boxes"])
by_pairs_other(dict([("boxes", boxes), ("labels", labels)])["labels"])
pairs = [("boxes", boxes), ("labels", labels)]
by_named_pairs(dict(pairs)["labels"])
by_tuple_of_pairs(dict((("boxes", boxes), ("labels", labels)))["boxes"])
by_copy(dict({"boxes": boxes, "labels": labels})["boxes"])
by_copy_other(dict({"boxes": boxes, "labels": labels})["labels"])
by_unpacking(dict(**{"boxes": boxes, "labels": labels})["boxes"])
base = {"boxes": boxes}
by_copy_and_keyword(dict(base, labels=labels)["boxes"])
by_copy_and_keyword_other(dict(base, labels=labels)["labels"])
by_keywords_get(dict(boxes=boxes, labels=labels).get("boxes"))
first = dict(value=boxes)
second = dict(value=labels)
by_first_call(first["value"])
by_second_call(second["value"])
by_function(build(boxes, labels)["boxes"])
empty = dict()
empty["boxes"] = boxes
by_empty_then_store(empty["boxes"])
by_parameter_key(pair("boxes", boxes)["boxes"])
by_items(dict({"boxes": boxes, "labels": labels}.items())["boxes"])
make = dict
by_alias(make(boxes=boxes, labels=labels)["boxes"])
by_nested_call(dict(inner=dict(boxes=boxes), labels=labels)["inner"]["boxes"])
by_copy_of_call(dict(dict(boxes=boxes, labels=labels))["boxes"])
by_copy_of_pairs(dict(dict([("boxes", boxes), ("labels", labels)]))["boxes"])
by_keyword_display(dict(inner={"boxes": boxes}, labels=labels)["inner"]["boxes"])
names = dict(key="boxes")
by_constant_value({"boxes": boxes, "labels": labels}[names["key"]])
merged = merge({"boxes": boxes}, {"labels": labels})
by_two_unpackings(merged["boxes"])
by_two_unpackings_other(merged["labels"])
