import issue127i

c = issue127i.C(42)
# Calling `c` will route to `call` because of the workaround in WALA-Cast-Python
a = c(val2=100)
