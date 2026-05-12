# Contributing

## Building

### Obtain Git Submodules

The following dependencies require Git submodules to be initialized and updated. If you cloned the repository without the `--recurse-submodules` option, you need to initialize and update the submodules:

```bash
git submodule update --init --recursive
```

**When adding or removing a Git submodule, update every place that hardcodes the submodule list to match `.gitmodules`.** Submodules are third-party code outside the Ariadne codebase under change, and several tools have to be told explicitly to skip them. Audit checklist:

- `pom.xml` — Spotless `<format>` blocks (search for `<!-- Exclude Git submodules -->`).
- `.github/workflows/continuous-integration.yml` — the `black --fast --check --extend-exclude ... .` invocation under "Check formatting with Black."
- `.github/codeql/codeql-config-java-kotlin.yml` — the `paths-ignore` block at the top.
- Install steps in `continuous-integration.yml` and `codeql.yml` — the build needs each submodule installed as a local Maven artifact (or otherwise made available), so renames/removals propagate here too.

To verify completeness, list current submodule paths and confirm each appears (or no longer appears) in every site above:

```bash
git config --file .gitmodules --get-regexp path | awk '{print $2}'
```

### Installing Jython 3

You must install the `jython-dev.jar` to your local maven repository.

1. Change directory to the cloned local Git repo: `cd jython3`.
1. Build Jython 3: `ant`. That will produce the file `jython3/dist/jython-dev.jar`.
1. Install the `jython-dev.jar` into your local maven repo (see [this post][SO post]):

	```bash
	mvn install:install-file \
	-Dfile=./jython-dev.jar \
	-DgroupId="org.python" \
	-DartifactId="jython3" \
	-Dversion="0.0.1-SNAPSHOT" \
	-Dpackaging="jar" \
	-DgeneratePom=true
	```
### Installing IDE

1. Change directory to `com.ibm.wala.cast.lsp`: `cd com.ibm.wala.cast.lsp`
1. Build and install to your local Maven repo: `mvn install`

### Building WALA/ML

Build and install to your local Maven repo: `mvn install`

## Commit Messages

- End commit messages with a period.
- Use backticks (fences) for code entities (e.g., class names, file names). For example: "Refactor `Input` definition in `tensorflow.xml` to correct package."

## Code Style

The repo ships a [pre-commit](https://pre-commit.com) configuration at `.pre-commit-config.yaml` that runs `mvn spotless:apply` and `black --fast` automatically before each commit. Install once after cloning:

```bash
pip install pre-commit && pre-commit install
```

After install, every `git commit` triggers `mvn spotless:apply` (which formats the whole repo) and `black --fast` (which formats touched Python files), so CI's `spotless:check` and `black --check` stay green. To run all hooks ad-hoc against the whole tree: `pre-commit run --all-files`.

If you prefer to run the formatters manually instead, the equivalent commands from the project root are:

```bash
mvn compile
mvn spotless:apply
black --fast .
```

## Python Testing

- Always verify that newly created Python test files run to completion using `python3.10`.
- Always add `assert` statements in Python test files for tensor `shape` and `dtype`.
- Ensure that `assert` statements in Python test files match the expectations defined in the corresponding JUnit test cases.

## Tensor Type Generators

When writing a new `TensorGenerator` subclass in `com.ibm.wala.cast.python.ml.client`, uphold the lattice conventions for shapes and dtypes so downstream analysis can distinguish "unknown tensor" (⊤) from "not a tensor" (⊥). The class-level Javadoc on `TensorGenerator` is the source of truth; the summary below is the quick version.

### Shapes — `getDefaultShapes`

| Return value | Meaning |
|---|---|
| `null` | ⊤ — the generator produces a tensor, but its shape cannot be determined. |
| empty set (`Collections.emptySet()`) | ⊥ — the variable is provably not a tensor. |
| non-empty set | The set of concrete shapes the tensor may take. |

Within a single shape, use `new SymbolicDim("?")` for a known-rank-but-unknown-size dimension (e.g., a dynamic batch size). A `null` shape *list* means even the rank is unknown.

### Dtypes — `getDefaultDTypes`

| Return value | Meaning |
|---|---|
| `EnumSet.of(DType.UNKNOWN)` | ⊤ — the generator produces a tensor, but its dtype cannot be determined. |
| empty set | ⊥ — the variable is provably not a tensor. |
| non-empty set of concrete `DType`s | The set of possible dtypes. |

Never return a bare empty set to mean "unknown dtype" — that collides with the "not a tensor" signal.

### Tensor Types — `getTensorTypes`

Shapes and dtypes are orthogonal. When the shape is unknown but the dtype is known, `getTensorTypes` emits `TensorType` instances with `null` dims so dtype information is preserved. `TensorType` is null-dims-safe; any code that consumes `TensorType`s must handle `getDims() == null`.

### Checklist When Adding a New Generator

- [ ] Audit every final-fallback return in `getDefaultShapes` and `getDefaultDTypes` against the tables above.
- [ ] Prefer `null` over `Collections.emptySet()` when the intended meaning is "we know it's a tensor, we just can't figure out the shape."
- [ ] Prefer `EnumSet.of(DType.UNKNOWN)` over `EnumSet.noneOf(DType.class)` / `Collections.emptySet()` for unknown dtypes.
- [ ] If your generator's result is accumulated into a `ret` set inside a loop, verify the final `return ret` cannot return an empty set when you actually meant "unknown." Add `return ret.isEmpty() ? null : ret;` if it can.

## Modeling APIs in `tensorflow.xml`

When adding or modifying an API summary in `tensorflow.xml` / `numpy.xml`:

- **Allocating ops** — APIs that return a fresh tensor (e.g., `tf.matmul`, `tf.nn.sigmoid`, `tf.nn.softmax`, `tf.sparse.add`, `tf.nn.sparse_softmax_cross_entropy_with_logits`, `np.array`, `np.reshape`) should use `<new def="res" class="..."/>` followed by `<return value="res"/>`. Pair the XML with a `TensorGenerator` subclass that computes the output shape/dtype from the inputs. **Never** use `<return value="param"/>` for an allocating op — that aliases the call's result with an input and silently propagates the wrong shape/dtype downstream (this is a recurring bug class; see closed wala/ML#412 and predecessors).
- **Allocation-class convention for `<new>` results** — picking the right `class="..."` depends on the API kind. The asymmetric rule is documented in the header comment at the top of `tensorflow.xml`; the short version: function-typed APIs that return a tensor (e.g., `tf.math.sigmoid`, `tf.math.add`, `tf.reshape`) use canonical `Ltensorflow/python/framework/ops/Tensor`; object-typed APIs and callback-bearing wrappers (e.g., `Variable`, `Estimator`, `SparseTensor`, Estimator's `train` method class) use the per-op class because class identity is load-bearing for the result's method surface or for virtual dispatch on stored callbacks. See wala/ML#459 (the migration that established this) and wala/ML#465 (the convention's documentation).
- **Genuinely pass-through ops** — only methods that semantically return one of their inputs unchanged (e.g., builder-style `Dataset.shuffle`/`batch` chain that returns the same dataset object, or identity-like ops) should use `<return value="param"/>`. Document the choice with a comment.
- **Allocatable classes** — `<new class="L..."/>` requires the target class to be declared `<class name="..." allocatable="true">` somewhere in the XML, otherwise `HeapModel.getInstanceKeyForAllocation` returns `null` and the iKey never reaches the caller (see wala/WALA#1889 history). When a new `<new>` target class is added, audit existing ones in the same area for the declaration.
- **`numArgs` / `paramNames` consistency** — every `<method numArgs="N" paramNames="...">` must have exactly `N` whitespace-separated names in `paramNames`. Mismatches silently break parameter resolution; audit when changing signatures.

## Pinning Shapes / Blocking PA Leaks

When the PA assignment graph propagates a tensor type into a destination that semantically isn't a tensor, use one of these `TensorTypeAnalysis` mechanisms (both threaded through `PythonTensorAnalysisEngine.performAnalysis`):

- **`setCalls`** (consumed as `set_shapes` in `TensorTypeAnalysis`) — pins a destination's shape to a specific value. Use for intentional shape overrides like `tf.set_shape`, slice-results, or subscript-results that need their receiver's shape blocked from leaking through (precedent: wala/ML#405).
- **`drops`** — pins a destination's tensor-type set to **empty + FIXED** via a `DropOp` edge transfer. Use for slots that are never tensors at runtime but get aliased to one through the PA graph (precedent: wala/ML#409 — `enumerate(...)`'s integer-index field).

## Java Testing

- Always run `mvn test` to ensure all Java tests pass before committing changes.
- Add JUnit test cases for any new functionality or bug fixes implemented in the Java codebase.
- Ensure that all new and existing JUnit test cases pass successfully before committing changes.
- Use descriptive names for JUnit test methods that clearly indicate the purpose of the test.
- If you change any of the summary files (e.g., `tensorflow.xml`), ensure that you add or update JUnit test cases to cover the changes made. Also, you must run `mvn clean` to ensure that the changes are correctly reflected in the build for summary (XML) files.
- When a test would fail because of a known precision/correctness gap, **prefer encoding the *observed* (current, imprecise) behavior in the assertion itself** with a `TODO(<issue>):` comment that names the precise post-fix form. When the fix lands, the actual result changes and the test starts failing with a clear "expected observed-form, got precise-form" diff — that's the cue to update the assertion. For example:

	```java
	/**
	 * ...test description, including why the result is currently imprecise...
	 *
	 * <p>TODO(wala/ML#380): When the per-Model collapse is fixed, narrow the assertion to
	 * {@code Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32)}.
	 */
	@Test
	public void testModelAttributesMultiModelWrapped() {
		test(..., Map.of(2, Set.of(TENSOR_64_5_FLOAT32, TENSOR_5_FLOAT32, TENSOR_64_7_FLOAT32, TENSOR_7_FLOAT32)));
	}
	```

	The fallback is `@Test(expected = AssertionError.class)`, which inverts the pass/fail signal — when the fix lands the test silently passes through an unrelated assertion (or fails because no `AssertionError` was thrown). Use it only when the precise post-fix shape is not yet known or not yet expressible in `TensorType`. When you do use it, always add a `TODO:` line to the test's Javadoc naming the blocking issue. For example:

	```java
	/**
	 * Test https://github.com/wala/ML/issues/210.
	 *
	 * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#210 is fixed.
	 */
	@Test(expected = AssertionError.class)
	public void testModule70() { ... }
	```

	Without the `TODO` keyword a suppressed failure is indistinguishable from an intentional positive negative-assertion (see `testDecoratedMethod9`), and IDE task-list / `grep TODO` tooling will not surface it as temporary. The only legitimate exception is when the `AssertionError` is itself the expected positive outcome (e.g., "this function doesn't exist, so the test should fail"); those should be documented in the Javadoc as intentional.

[SO post]: https://stackoverflow.com/questions/4955635/how-to-add-local-jar-files-to-a-maven-project#answer-4955635
