# Test https://github.com/wala/ML/issues/1339: a `get` with a default, on a dictionary that
# `json.loads` returns, and a `get` on the value that yields, are unknown to the analysis.
import json

import tensorflow as tf


def consume_rows(t):
    assert t.shape == (5, 3)


def main(text):
    config = json.loads(text) if text else {}
    rows = config.get("shape", {}).get("rows", 2)
    consume_rows(tf.ones([rows, 3]))


main('{"shape": {"rows": 5}}')
