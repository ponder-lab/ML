# The PYTHONPATH is the project root then `src` inside it, and both hold `pkg/sub/mod.py`. Python
# binds `pkg.sub.mod` to the root's, the first entry's, and `src/pkg/sub/mod.py` is shadowed for
# that name, though it is still code of the project (wala/ML#984).
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import tensorflow as tf
from pkg.sub.mod import take


def consume(x):
    assert x.shape == (3,)
    assert x.dtype == tf.float32


consume(take())
