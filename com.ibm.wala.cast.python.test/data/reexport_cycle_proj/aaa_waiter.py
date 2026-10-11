# Sorts before the package, and waits behind the cycle: its base comes from `mod.py`, which is on
# it, but this module is not.
from cyc.mod import Child


class Waiter(Child):
    def __init__(self):
        super().__init__()
