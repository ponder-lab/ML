# Test https://github.com/wala/ML/issues/1177: `N` has no `__init__.py`, so it is an implicit
# namespace package, and its submodules are imported through it.
from tensorflow import ones
from N import B
import N.D.E
from N.P import Q

B.f(ones([1, 2]))
N.D.E.g(ones([1, 2]))
Q.h(ones([1, 2]))
