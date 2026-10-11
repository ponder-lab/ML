import pkg
from pkg import inputters
from pkg.inputters import ExampleInputter
from pkg.inputters.inputter import ExampleInputter as DefiningModuleBase


class ViaPackageAttribute(inputters.ExampleInputter):
    def __init__(self, features_inputter):
        super().__init__(features_inputter)


class ViaPackageName(ExampleInputter):
    def __init__(self, features_inputter):
        super().__init__(features_inputter)


class ViaTwoPackages(pkg.ExampleInputter):
    def __init__(self, features_inputter):
        super().__init__(features_inputter)


class ViaRelativeReexport(inputters.Inputter):
    pass


class ViaDefiningModule(DefiningModuleBase):
    def __init__(self, features_inputter):
        super().__init__(features_inputter)
