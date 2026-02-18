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
- PythonTensorAnalysisEngine now handles SSANewInstruction for Ltensorflow/python/framework/constant_op/constant to support inlined tf.constant.
- Iterating CallGraph in getDataflowSources is more robust for finding synthetic allocations (like tf.constant) than iterating the dataflow graph.
- Updated testAdd3, testAdd6, testMultiply expectations to match actual behavior (TENSOR_5_INT32/TENSOR_4_INT32 instead of MNIST_INPUT).
- RaggedFromNestedValueRowIds generator updated to recursively extract values from CONSTANT_OP_CONSTANT objects to support inlined tf.constant.
- To expose implicit allocations (like `tf.constant`) to the static analysis, modify the XML summary to call a synthetic method (e.g., `read_data`) on the allocation. This forces the creation of a trackable dataflow source.
- When modeling shape-transforming operations (like `reduce_mean`) in `tensorflow.xml`, ensure the operation returns a new object (e.g., via `<new ... />`) instead of returning the input tensor. This prevents the input shape from aliasing with the output, allowing the `TensorGenerator` to provide the sole, correct shape.
- In `TensorGenerator` implementations, use `getArgumentValueNumber(builder, paramIndex)` to resolve parameter indices to SSA value numbers, and handle numeric constant arguments as `Number` (checking for both `Integer` and `Long`) to support different internal representations of Python integers.
- Prefer numbered test names (e.g., testReduceMean4) over descriptive names for additional tests of the same operation.
- When defining a class in `tensorflow.xml` with `allocatable="true"`, if you use `<call name="read_data" ... />` in the `do` method, you MUST explicitly define the `read_data` method in the class (usually returning a `<new ... />`), otherwise `XMLMethodSummaryReader` may fail with "bad xml file". Also, the `call` tag should usually specify `class="LRoot"` (or the appropriate class) if calling a synthetic method on `self` where `self` is treated as `LRoot` or similar.
- Use backticks for code entities in Git commit messages.
- When resolving DTypes in `TensorGenerator`, always check against known DType fields (like `tf.float32`) even if the instance type is `LRoot` or generic, to handle imprecise type inference. Also, do not rely on `getArgumentValueNumber` returning a valid value number in `getDTypes` (or similar) if the parameter might be unused and optimized away in the callee IR; rely on `getArgumentPointsToSet` which checks callers.
- The failure in `TestTensorflow2Model.testMultiply7` is caused by aliasing of `tf.constant` results due to 1-CFA context insensitivity merging calls to the helper `read_data` method. Increasing context depth to 2 (2-CFA) fixes this but is considered too expensive. The preferred fix is to refactor the analysis engine to support isolated allocations in `do` without the `read_data` helper.
- The `read_data` helper method pattern in `tensorflow.xml` is deprecated and has been replaced by direct allocations (e.g., `<new def="x" ... />`) within the `do` method of operations. This ensures correct 1-CFA context sensitivity and prevents aliasing. `TensorGenerator` now supports shape/dtype inference for these direct allocations.
- When modeling factory methods in `tensorflow.xml` (like `Input`, `constant`, `Tensor`), inline the `<new ... />` allocation directly into the `do` method. Do not use a helper `read_data` method. This ensures that 1-CFA call-string context sensitivity correctly distinguishes separate calls to the factory from the user script, preventing aliasing of the resulting tensor objects.
- When WALA's `PropagationSystem.findOrCreatePointsToSet` throws `UnimplementedError` (often due to implicit PointerKeys in synthetic methods), a manual `TensorGenerator` instantiation can be used as a fallback, overriding methods like `getArgumentPointsToSet`, `getShapes`, and `getDTypes` to access IR directly and avoid `source`-based lookups. Also, ensure logging guards against null `source`.
- Use `createManualGenerator` in `TensorGenerator.java` to handle synthetic operations (like `tf.ones`, `tf.sparse.eye`) where inlined allocations cause WALA to throw `UnimplementedError` due to implicit pointer keys. Ensure the manual generator overrides `getArgumentPointsToSet`, `getShapes`, `getDTypes`, and `toString` to bypass standard `source`-dependent logic.
