# Regression pin for wala/ML#816: an array allocated with a complex dtype argument resolves to that
# complex dtype, not to numpy's float64 default. Asserting complex64/complex128 positively is what
# makes this catch the original defect: if the dtype argument were disregarded and the allocator
# fell back to float64 again, these assertions fail rather than pass.
import numpy as np


def consume_complex64(x):
    pass


def consume_complex128(x):
    pass


c64 = np.zeros((4, 5), dtype=np.complex64)
assert c64.dtype == np.complex64 and c64.shape == (4, 5)
consume_complex64(c64)

c128 = np.ones((2, 3), dtype=np.complex128)
assert c128.dtype == np.complex128 and c128.shape == (2, 3)
consume_complex128(c128)
