import issue127d

x = issue127d.C()

# Direct call
a = x.foo(5)

# Call through a variable
y = x.foo
b = y(3)
