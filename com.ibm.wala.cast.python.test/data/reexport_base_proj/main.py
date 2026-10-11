import tensorflow as tf

from pkg import inputters
from pkg.models import s2s


def consume_defining_module(x):
    assert x.dtype == tf.float32
    assert x.shape == (2, 3)


def consume_package_attribute(x):
    assert x.dtype == tf.float32
    assert x.shape == (2, 3)


def consume_package_name(x):
    assert x.dtype == tf.float32
    assert x.shape == (2, 3)


def consume_two_packages(x):
    assert x.dtype == tf.float32
    assert x.shape == (2, 3)


def consume_layer_call(x):
    assert x.dtype == tf.float32
    assert x.shape == (2, 3)


def consume_relative_reexport(x):
    assert x.dtype == tf.float32
    assert x.shape == (2, 3)


v = tf.ones((2, 3))
consume_defining_module(
    s2s.ViaDefiningModule(inputters.WordEmbedder()).features_inputter(v)
)
consume_package_attribute(
    s2s.ViaPackageAttribute(inputters.WordEmbedder()).features_inputter(v)
)
consume_package_name(s2s.ViaPackageName(inputters.WordEmbedder()).features_inputter(v))
consume_two_packages(s2s.ViaTwoPackages(inputters.WordEmbedder()).features_inputter(v))
consume_layer_call(s2s.ViaPackageAttribute(inputters.WordEmbedder())(v))
consume_relative_reexport(s2s.ViaRelativeReexport()(v))
