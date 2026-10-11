# A vendored copy whose module path, `pkg/inputters`, collides with the project's own package; its
# re-exported base sets no inputter, so a base resolved here instead of in the project's own
# package leaves the subclass's field unset.
from pkg.inputters.inputter import ExampleInputter
