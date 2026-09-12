import sys

import tensorflow as tf
from tensorflow.keras import applications

# The output rank of a `tf.keras.applications` model is fixed by its constructor (wala/ML#896):
# `include_top` (default True) classifies to rank 2, `pooling` of "avg" or "max" reduces the feature
# map to rank 2, and `include_top=False` with no pooling leaves the rank-4 feature map. The extents
# other than the batch are the architecture's own and stay unresolved. Every model is `weights=None`
# so the program runs anywhere; the shapes do not depend on weights.


def consume_top(t):
    pass


def consume_avg(a):
    pass


def consume_max(m):
    pass


def consume_features(f):
    pass


def consume_flag(g):
    pass


def consume_disagree(d):
    pass


def consume_looped(l):
    pass


x = tf.ones((4, 224, 224, 3))

top = applications.MobileNetV2(weights=None)(x)
assert top.shape.as_list() == [4, 1000], top.shape
consume_top(top)

avg = applications.mobilenet_v2.MobileNetV2(
    include_top=False, weights=None, pooling="avg"
)(x)
assert avg.shape.as_list() == [4, 1280], avg.shape
consume_avg(avg)

mx = applications.MobileNetV2(include_top=False, weights=None, pooling="max")(x)
assert mx.shape.as_list() == [4, 1280], mx.shape
consume_max(mx)

features = applications.EfficientNetB0(include_top=False, weights=None)(x)
assert features.shape.as_list() == [4, 7, 7, 1280], features.shape
consume_features(features)

# An `include_top` the program decides at runtime: the analysis cannot pick a rank and declines.
flag = len(sys.argv) > 5
mixed = applications.MobileNetV2(include_top=flag, weights=None)(x)
assert mixed.shape.as_list() == ([4, 1000] if flag else [4, 7, 7, 1280]), mixed.shape
consume_flag(mixed)

# Two constructions with different `include_top` reaching one call: each is its own instance, so
# the call's type is the union of both ranks, the path-insensitive reading of the branch.
model = (
    applications.MobileNetV2(weights=None)
    if flag
    else applications.MobileNetV2(include_top=False, weights=None)
)
either = model(x)
assert either.shape.as_list() == ([4, 1000] if flag else [4, 7, 7, 1280]), either.shape
consume_disagree(either)

# One construction site with both `include_top` values: the stored argument holds two constants,
# so the analysis declines rather than pick either.
for top in (True, False):
    looped = applications.MobileNetV2(include_top=top, weights=None)(x)
    assert looped.shape.as_list() == (
        [4, 1000] if top else [4, 7, 7, 1280]
    ), looped.shape
    consume_looped(looped)
