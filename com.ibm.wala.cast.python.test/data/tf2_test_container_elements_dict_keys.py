# Test `ContainerElements` on dicts keyed by constants other than ints and strings, by `None`, and
# by a key the analysis cannot resolve. Each sink receives one dict.
import json

import tensorflow as tf

x = tf.ones([2, 3])


def sink_float_tensor(c):
    assert len(c) == 1 and c[1.5] is x


def sink_float_none(c):
    assert len(c) == 1 and c[1.5] is None


def sink_bool_tensor(c):
    assert len(c) == 1 and c[True] is x


def sink_bool_none(c):
    assert len(c) == 1 and c[True] is None


def sink_none_key(c):
    assert len(c) == 1 and c[None] is None


def sink_unresolved_key(c):
    assert len(c) == 1 and c["k"] is None


sink_float_tensor({1.5: x})
sink_float_none({1.5: None})
sink_bool_tensor({True: x})
sink_bool_none({True: None})
sink_none_key({None: None})

unresolved = {}
k = json.loads('"k"')
unresolved[k] = None
sink_unresolved_key(unresolved)
