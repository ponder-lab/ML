# Test https://github.com/wala/ML/issues/163.

from tensorflow import ones
from C.D import B
from C import E

B.f(ones([1, 2]))
E.g(ones([1, 2]))
