# Contributing

## Building

### Obtain Git Submodules

The following dependencies require Git submodules to be initialized and updated. If you cloned the repository without the `--recurse-submodules` option, you need to initialize and update the submodules:

```bash
git submodule update --init --recursive
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

Please run the following commands from the project root directory before committing any changes to ensure code style consistency and that the project compiles:

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

### Tensor types — `getTensorTypes`

Shapes and dtypes are orthogonal. When the shape is unknown but the dtype is known, `getTensorTypes` emits `TensorType` instances with `null` dims so dtype information is preserved. `TensorType` is null-dims-safe; any code that consumes `TensorType`s must handle `getDims() == null`.

### Checklist when adding a new generator

- [ ] Audit every final-fallback return in `getDefaultShapes` and `getDefaultDTypes` against the tables above.
- [ ] Prefer `null` over `Collections.emptySet()` when the intended meaning is "we know it's a tensor, we just can't figure out the shape."
- [ ] Prefer `EnumSet.of(DType.UNKNOWN)` over `EnumSet.noneOf(DType.class)` / `Collections.emptySet()` for unknown dtypes.
- [ ] If your generator's result is accumulated into a `ret` set inside a loop, verify the final `return ret` cannot return an empty set when you actually meant "unknown." Add `return ret.isEmpty() ? null : ret;` if it can.

## Java Testing

- Always run `mvn test` to ensure all Java tests pass before committing changes.
- Add JUnit test cases for any new functionality or bug fixes implemented in the Java codebase.
- Ensure that all new and existing JUnit test cases pass successfully before committing changes.
- Use descriptive names for JUnit test methods that clearly indicate the purpose of the test.
- If you change any of the summary files (e.g., `tensorflow.xml`), ensure that you add or update JUnit test cases to cover the changes made. Also, you must run `mvn clean` to ensure that the changes are correctly reflected in the build for summary (XML) files.
- When suppressing a known-failing test with `@Test(expected = AssertionError.class)`, always add a `TODO:` line to the test's Javadoc naming the issue that needs to land before the annotation can be removed. For example:

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
