# Test for a src layout whose package the test imports by a dotted name, as the `src` PYTHONPATH
# entry makes it importable, and reaches through the package's initializers.
import tensorflow as tf

import pkg.sub


class TestPkg:
    def test_take(self):
        pkg.sub.take(tf.ones((4, 5)))
