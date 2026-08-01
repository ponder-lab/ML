import numpy as np


def consume_int_scaled(s):
    pass


def consume_float_scaled(f):
    pass


# NumPy promotes an integer array divided by a Python float to float64, not float32.
# This is the ubiquitous image-normalization idiom.
x = np.zeros((4, 5), dtype=np.uint8)
scaled = x / 255.0
assert scaled.dtype == np.float64 and scaled.shape == (4, 5)
consume_int_scaled(scaled)

# A float32 array keeps float32 under the same operation, so the promotion is not
# "any float literal makes the result float64" either.
y = np.zeros((4, 5), dtype=np.float32)
float_scaled = y / 255.0
assert float_scaled.dtype == np.float32 and float_scaled.shape == (4, 5)
consume_float_scaled(float_scaled)


def consume_complex_scaled(c):
    pass


# A complex array keeps its own dtype too, so the rule is "an integral operand widens to
# float64, every other numeric operand keeps itself" rather than anything about the literal.
z = np.zeros((4, 5), dtype=np.complex64)

complex_scaled = z / 255.0
assert complex_scaled.dtype == np.complex64 and complex_scaled.shape == (4, 5)
consume_complex_scaled(complex_scaled)
