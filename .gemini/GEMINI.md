## Gemini Added Memories
- For Python tests:
	- Verify they run to completion using `python3.10`.
	- Add `assert` statements for tensor shapes and dtypes.
	- Ensure these `assert` statements match corresponding JUnit test case expectations.
- For TensorFlow operations, always process and test positional, keyword, and mixed arguments.
- Always run `mvn clean install` first whenever a test cannot find a test file, or after modifying `tensorflow.xml`.
- Fixed `testEye6` regression where `tf.constant` result shape was not inferred. Root causes:
1. Missing class definition for `Ltensorflow/python/framework/constant_op/constant` in `tensorflow.xml`.
2. `constant` function model didn't store `value` in the result object.
3. `TensorGenerator.java` didn't handle `CONSTANT_OP_CONSTANT` to extract the value for shape analysis.
Fix involved defining the class, updating the model to `putfield` the value, and updating `TensorGenerator` to recurse into the `value` field using `getShapesFromShapeArgument`.
- `PythonTensorAnalysisEngine` now handles `SSANewInstruction` for `Ltensorflow/python/framework/constant_op/constant` to support inlined `tf.constant`.
- Iterating `CallGraph` in `getDataflowSources` is more robust for finding synthetic allocations (like tf.constant) than iterating the dataflow graph.
- Updated `testAdd3`, `testAdd6`, `testMultiply` expectations to match actual behavior (`TENSOR_5_INT32`/`TENSOR_4_INT32` instead of `MNIST_INPUT`).
- `RaggedFromNestedValueRowIds` generator updated to recursively extract values from `CONSTANT_OP_CONSTANT` objects to support inlined `tf.constant`.
- To expose implicit allocations (like `tf.constant`) to the static analysis, modify the XML summary to call a synthetic method (e.g., `read_data`) on the allocation. This forces the creation of a trackable dataflow source.
- When modeling shape-transforming operations (like `reduce_mean`) in `tensorflow.xml`, ensure the operation returns a new object (e.g., via `<new ... />`) instead of returning the input tensor. This prevents the input shape from aliasing with the output, allowing the `TensorGenerator` to provide the sole, correct shape.
- Prefer numbered test names (e.g., testReduceMean4) over descriptive names for additional tests of the same operation.
- When defining a class in `tensorflow.xml` with `allocatable="true"`, if you use `<call name="read_data" ... />` in the `do` method, you MUST explicitly define the `read_data` method in the class (usually returning a `<new ... />`), otherwise `XMLMethodSummaryReader` may fail with "bad xml file". Also, the `call` tag should usually specify `class="LRoot"` (or the appropriate class) if calling a synthetic method on `self` where `self` is treated as `LRoot` or similar.
- Use backticks for code entities in Git commit messages.
- When resolving DTypes in `TensorGenerator`, always check against known DType fields (like `tf.float32`) even if the instance type is `LRoot` or generic, to handle imprecise type inference. Also, do not rely on `getArgumentValueNumber` returning a valid value number in `getDTypes` (or similar) if the parameter might be unused and optimized away in the callee IR; rely on `getArgumentPointsToSet` which checks callers.
- The failure in `TestTensorflow2Model.testMultiply7` is caused by aliasing of `tf.constant` results due to 1-CFA context insensitivity merging calls to the helper `read_data` method. Increasing context depth to 2 (2-CFA) fixes this but is considered too expensive. The preferred fix is to refactor the analysis engine to support isolated allocations in `do` without the `read_data` helper.
- The `read_data` helper method pattern in `tensorflow.xml` is deprecated and has been replaced by direct allocations (e.g., `<new def="x" ... />`) within the `do` method of operations. This ensures correct 1-CFA context sensitivity and prevents aliasing. `TensorGenerator` now supports shape/dtype inference for these direct allocations.
- When modeling factory methods in `tensorflow.xml` (like `Input`, `constant`, `Tensor`), inline the `<new ... />` allocation directly into the `do` method. Do not use a helper `read_data` method. This ensures that 1-CFA call-string context sensitivity correctly distinguishes separate calls to the factory from the user script, preventing aliasing of the resulting tensor objects.
- When logging exceptions in Java, prefer `LOGGER.log(Level.FINE, message, exception)` over `LOGGER.fine(message + exception.getMessage())` to preserve stack traces. Also ensure `java.util.logging.Level` is imported.
- In this multi-module Maven project, when modifying the main `ml` module and running tests in `ml.test` module, `mvn compile` on the root might not be sufficient for `ml.test` to pick up changes if `ml.test` relies on the installed artifact. Use `mvn install -pl com.ibm.wala.cast.python.ml -am -DskipTests` or `mvn clean compile` to ensure changes are propagated.
- Always redirect test output to a file (e.g., `> test_output.txt`) to avoid overwhelming the console and to facilitate easier analysis of the results.
- Always check the complete test suite for regressions running `mvn clean install` from the project root directory after making changes to ensure that no unintended side effects have been introduced.
- `TensorGenerator` shape/dtype lattice convention (see class Javadoc and `CONTRIBUTING.md`): `getDefaultShapes` returns `null` for unknown shape (⊤), empty set for not-a-tensor (⊥), non-empty set for concrete shapes. `getDefaultDTypes` returns `EnumSet.of(DType.UNKNOWN)` for unknown dtype, empty set for not-a-tensor, non-empty set of `DType`s otherwise. `SymbolicDim("?")` is for known-rank-but-unknown-size dims within a shape — distinct from a `null` shape list, which means even the rank is unknown. `TensorType` is null-dims-safe; `getTensorTypes` emits `TensorType(dtype, null)` when shape is unknown but dtype is known. New `TensorGenerator` subclasses must uphold all of this.
