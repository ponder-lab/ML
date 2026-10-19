# A root-level script beside a `src` layout, so the PYTHONPATH holds the project root and `src`
# inside it. The package's own imports name it `pkg`, as the `src` entry makes it importable
# (wala/ML#984).
import os
import sys

sys.path.insert(1, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import tensorflow as tf

import pkg.sub

pkg.sub.take(tf.ones((4, 5)))
