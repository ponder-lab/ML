# Contributing

## Building

### Obtain Git Submodules

The following dependencies require Git submodules to be initialized and updated. If you cloned the repository without the `--recurse-submodules` option, you need to initialize and update the submodules:

```bash
git submodule update --init --recursive
```

**When adding or removing a Git submodule, update every place that hardcodes the submodule list to match `.gitmodules`.** Submodules are third-party code outside the Ariadne codebase under change, and several tools have to be told explicitly to skip them. Audit checklist:

- `pom.xml`—Spotless `<format>` blocks (search for `<!-- Exclude Git submodules -->`).
- `.github/workflows/continuous-integration.yml`—the `black --fast --check --extend-exclude ... .` invocation under "Check formatting with Black."
- `.github/codeql/codeql-config-java-kotlin.yml`—the `paths-ignore` block at the top.
- Install steps in `continuous-integration.yml` and `codeql.yml`—the build needs each submodule installed as a local Maven artifact (or otherwise made available), so renames/removals propagate here too.

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
	-Dversion="0.0.2" \
	-Dpackaging="jar" \
	-DgeneratePom=true
	```
### Installing IDE

1. Change directory to `IDE/com.ibm.wala.cast.lsp` (the IDE code lives in a Git submodule at `IDE/`, per `.gitmodules`): `cd IDE/com.ibm.wala.cast.lsp`
1. Build (without installing): `mvn package -DskipTests`
1. Install at the non-SNAPSHOT coordinate `0.0.1` (the `cast.lsp` POM declares `0.0.1-SNAPSHOT`, but the release-plugin rejects SNAPSHOT deps; we install under a release coordinate locally so Ariadne's `release:prepare` runs in batch mode without `-DignoreSnapshots`):

	```bash
	mvn install:install-file \
	-Dfile=target/com.ibm.wala.cast.lsp-0.0.1-SNAPSHOT.jar \
	-DgroupId="com.ibm.wala" \
	-DartifactId="com.ibm.wala.cast.lsp" \
	-Dversion="0.0.1" \
	-Dpackaging="jar" \
	-DgeneratePom=true
	```

### Building WALA/ML

Build and install to your local Maven repo: `mvn install`

## Cutting a Release

Releases are published to GitHub Packages (`maven.pkg.github.com/ponder-lab/ML`), including the consumer-facing fat JAR `com.ibm.wala.cast.python.ml-X.Y.Z-fat.jar`. Cutting a release uses `maven-release-plugin`; CI publishes the artifacts automatically when the release tag is pushed, so there is no manual `mvn deploy` step.

### Prerequisites

- The local-dependency installs above (Jython 3 at `0.0.2` and `cast.lsp` at `0.0.1`). `release:prepare` rejects SNAPSHOT dependencies, so these must be present at their release coordinates.
- Push access to `master`. `release:prepare` pushes two version-bump commits and the tag directly to `master`, bypassing the pull-request requirement via the OrganizationAdmin bypass (sanctioned per [wala/ML#457](https://github.com/wala/ML/issues/457)).
- A classic PAT is *not* needed for the standard flow: the operator only pushes git commits and a tag; CI does the GitHub Packages deploy. (A classic PAT with `write:packages` is only needed for the manual-deploy fallback below — fine-grained PATs do not authenticate to `maven.pkg.github.com`.)

### Steps

1. From a clean checkout of `master`, run:

	```bash
	mvn release:clean -B
	mvn release:prepare -B \
		-DreleaseVersion=X.Y.Z \
		-DdevelopmentVersion=X.Y.W-SNAPSHOT
	```

	For example, to cut `0.48.0` with the next development version `0.48.1-SNAPSHOT`, use `-DreleaseVersion=0.48.0 -DdevelopmentVersion=0.48.1-SNAPSHOT`. No other flags are needed: the tag format is pinned to plain `X.Y.Z` ([wala/ML#560](https://github.com/wala/ML/issues/560)) and the local dependencies are installed at release coordinates (see the install steps above), so `-Dtag`, `-DignoreSnapshots`, and `-DallowTimestampedSnapshots` are unnecessary. `release:prepare` builds and tests the release version, sets it, commits, tags, sets the next development version, commits, and pushes everything to `master`.

1. The tag push triggers CI's deploy gate ([wala/ML#421](https://github.com/wala/ML/issues/421)), which builds the tag and publishes the artifacts to GitHub Packages. The gate requires the tag to be plain semver *and* an ancestor of `master`.

1. Publish the GitHub release. Convention is `--prerelease`, and the notes should reflect the full diff since the previous release (`git log <prev-tag>..X.Y.Z`), not just one change:

	```bash
	gh release create X.Y.Z --prerelease --title "X.Y.Z" --notes "..."
	```

1. Confirm the version published:

	```bash
	gh api "/orgs/ponder-lab/packages/maven/com.ibm.wala.com.ibm.wala.cast.python.ml/versions" --jq '.[].name' | head
	```

### Troubleshooting

- **The tag pushed but the deploy did not fire** (rare — e.g., the tag's commit predates a workflow fix). Re-trigger the tag-push event: `git push --delete origin X.Y.Z && git push origin X.Y.Z`. ([wala/ML#454](https://github.com/wala/ML/issues/454) tracks a `workflow_dispatch` recovery for this.)
- **The pre-commit hook rejects `release:prepare`'s commit** ("files were modified by this hook"). This was a Spotless / `maven-release-plugin` self-closing-tag conflict, fixed in [wala/ML#566](https://github.com/wala/ML/issues/566) (`sortPom`'s `spaceBeforeCloseEmptyElement`). If it ever recurs, run the release with git hooks disabled: `git config core.hooksPath "$(mktemp -d)"`, run `release:prepare`, then `git config --unset core.hooksPath`.
- **Manual-deploy fallback.** If CI cannot deploy a tag at all, deploy from a clean local build: check out the release commit, `mvn spotless:apply` (working tree only), then `mvn clean deploy -DskipTests -B` using a `~/.m2/settings.xml` `github` server with a classic `write:packages` PAT.

### One-Click Releases (Planned)

[wala/ML#455](https://github.com/wala/ML/issues/455) tracks moving this to a `workflow_dispatch` GitHub Action so a release can be cut from the Actions UI with no local toolchain or credentials.

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
- Keep `assert` statements in Python test files in sync with the corresponding JUnit test cases—in both directions. Mismatches between the two are a red flag.
- For full DL training scripts (e.g., `tensorflow_gan_tutorial.py`), running to completion with `python3.10` may be impractical. Run with a short timeout, confirm no `AssertionError`, and document the limitation in the JUnit Javadoc or commit message.

## Tensor Type Generators

When writing a new `TensorGenerator` subclass in `com.ibm.wala.cast.python.ml.client`, uphold the lattice conventions for shapes and dtypes so downstream analysis can distinguish "unknown tensor" (⊤) from "not a tensor" (⊥). The class-level Javadoc on `TensorGenerator` is the source of truth; the summary below is the quick version.

### Shapes—`getDefaultShapes`

| Return value | Meaning |
|---|---|
| `null` | ⊤—the generator produces a tensor, but its shape cannot be determined. |
| empty set (`Collections.emptySet()`) | ⊥—the variable is provably not a tensor. |
| non-empty set | The set of concrete shapes the tensor may take. |

Within a single shape, use `new SymbolicDim("?")` for a known-rank-but-unknown-size dimension (e.g., a dynamic batch size). A `null` shape *list* means even the rank is unknown.

### Dtypes—`getDefaultDTypes`

| Return value | Meaning |
|---|---|
| `EnumSet.of(DType.UNKNOWN)` | ⊤—the generator produces a tensor, but its dtype cannot be determined. |
| empty set | ⊥—the variable is provably not a tensor. |
| non-empty set of concrete `DType`s | The set of possible dtypes. |

Never return a bare empty set to mean "unknown dtype"—that collides with the "not a tensor" signal.

### Tensor Types—`getTensorTypes`

Shapes and dtypes are orthogonal. When the shape is unknown but the dtype is known, `getTensorTypes` emits `TensorType` instances with `null` dims so dtype information is preserved. `TensorType` is null-dims-safe; any code that consumes `TensorType`s must handle `getDims() == null`.

The per-variable result aggregated from the per-axis lattice tables above follows its own lattice on the returned `Set<TensorType>`:

| Return value | Meaning |
|---|---|
| `null` | Defensive fallback for contract violation (dtype set is `null`/empty rather than `EnumSet.of(DType.UNKNOWN)`, and shape is also `null`). Under contract-compliant generators this case shouldn't occur. |
| empty set | ⊥—the variable is provably not a tensor. |
| non-empty set | The variable has at least one possible `TensorType`. ⊤ at the variable level (known to be a tensor, shape and dtype unknown) is encoded here too, as a non-empty Set containing `TensorType(UNKNOWN, null)`. Individual elements may still carry `getDims() == null` (shape-⊤) or `getDType() == DType.UNKNOWN` (dtype-⊤). |

Downstream consumers iterating aggregated state (e.g., `TensorTypeAnalysis.iterator()`) filter to `state != null && !state.isEmpty()`. Under contract-compliant generators this filters out ⊥ (empty Set); ⊤ at the variable level remains visible to the iterator as a non-empty Set with `TensorType(UNKNOWN, null)` inside. A non-empty entry may carry shape-⊤ or dtype-⊤ on individual `TensorType` instances, which consumers must handle.

### Checklist When Adding a New Generator

- [ ] Audit every final-fallback return in `getDefaultShapes` and `getDefaultDTypes` against the tables above.
- [ ] Prefer `null` over `Collections.emptySet()` when the intended meaning is "we know it's a tensor, we just can't figure out the shape."
- [ ] Prefer `EnumSet.of(DType.UNKNOWN)` over `EnumSet.noneOf(DType.class)`/`Collections.emptySet()` for unknown dtypes.
- [ ] If your generator's result is accumulated into a `ret` set inside a loop, verify the final `return ret` cannot return an empty set when you actually meant "unknown." Add `return ret.isEmpty() ? null : ret;` if it can.

## Modeling APIs in `tensorflow.xml`

When adding or modifying an API summary in `tensorflow.xml`/`numpy.xml`:

- **Allocating ops**—APIs that return a fresh tensor (e.g., `tf.matmul`, `tf.nn.sigmoid`, `tf.nn.softmax`, `tf.sparse.add`, `tf.nn.sparse_softmax_cross_entropy_with_logits`, `np.array`, `np.reshape`) should use `<new def="res" class="..."/>` followed by `<return value="res"/>`. Pair the XML with a `TensorGenerator` subclass that computes the output shape/dtype from the inputs. **Never** use `<return value="param"/>` for an allocating op—that aliases the call's result with an input and silently propagates the wrong shape/dtype downstream (this is a recurring bug class; see closed wala/ML#412 and predecessors).
- **Allocation-class convention for `<new>` results**—picking the right `class="..."` depends on the API kind. The asymmetric rule is documented in the header comment at the top of `tensorflow.xml`; the short version: function-typed APIs that return a tensor (e.g., `tf.math.sigmoid`, `tf.math.add`, `tf.reshape`) use canonical `Ltensorflow/python/framework/ops/Tensor`; object-typed APIs and callback-bearing wrappers (e.g., `Variable`, `Estimator`, `SparseTensor`, Estimator's `train` method class) use the per-op class because class identity is load-bearing for the result's method surface or for virtual dispatch on stored callbacks. See wala/ML#459 (the migration that established this) and wala/ML#465 (the convention's documentation).
- **Genuinely pass-through ops**—only methods that semantically return one of their inputs unchanged (e.g., builder-style `Dataset.shuffle`/`batch` chain that returns the same dataset object, or identity-like ops) should use `<return value="param"/>`. Document the choice with a comment.
- **Allocatable classes**—`<new class="L..."/>` requires the target class to be declared `<class name="..." allocatable="true">` somewhere in the XML, otherwise `HeapModel.getInstanceKeyForAllocation` returns `null` and the iKey never reaches the caller (see wala/WALA#1889 history). When a new `<new>` target class is added, audit existing ones in the same area for the declaration.
- **`numArgs`/`paramNames` consistency**—every `<method numArgs="N" paramNames="...">` must have exactly `N` whitespace-separated names in `paramNames`. Mismatches silently break parameter resolution; audit when changing signatures.

## Pinning Shapes/Blocking PA Leaks

When the PA assignment graph propagates a tensor type into a destination that semantically isn't a tensor, use one of these `TensorTypeAnalysis` mechanisms (both threaded through `PythonTensorAnalysisEngine.performAnalysis`):

- **`setCalls`** (consumed as `set_shapes` in `TensorTypeAnalysis`)—pins a destination's shape to a specific value. Use for intentional shape overrides like `tf.set_shape`, slice-results, or subscript-results that need their receiver's shape blocked from leaking through (precedent: wala/ML#405).
- **`drops`**—pins a destination's tensor-type set to **empty + FIXED** via a `DropOp` edge transfer. Use for slots that are never tensors at runtime but get aliased to one through the PA graph (precedent: wala/ML#409—`enumerate(...)`'s integer-index field).

## Java Testing

- Always run `mvn test` to ensure all Java tests pass before committing changes.
- Add JUnit test cases for any new functionality or bug fixes implemented in the Java codebase.
- Ensure that all new and existing JUnit test cases pass successfully before committing changes.
- Use descriptive names for JUnit test methods that clearly indicate the purpose of the test.
- If you change any of the summary files (e.g., `tensorflow.xml`), ensure that you add or update JUnit test cases to cover the changes made. Also, you must run `mvn clean` to ensure that the changes are correctly reflected in the build for summary (XML) files.
- When a test would fail because of a known precision/correctness gap, **prefer encoding the *observed* (current, imprecise) behavior in the assertion itself** with a `TODO(<issue>):` comment that names the precise post-fix form. When the fix lands, the actual result changes and the test starts failing with a clear "expected observed-form, got precise-form" diff—that's the cue to update the assertion. For example:

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

	The fallback is `@Test(expected = AssertionError.class)`, which inverts the pass/fail signal—when the fix lands the test silently passes through an unrelated assertion (or fails because no `AssertionError` was thrown). Use it only when the precise post-fix shape is not yet known or not yet expressible in `TensorType`. When you do use it, always add a `TODO:` line to the test's Javadoc naming the blocking issue. For example:

	```java
	/**
	 * Test https://github.com/wala/ML/issues/210.
	 *
	 * <p>TODO: Remove {@code expected = AssertionError.class} once wala/ML#210 is fixed.
	 */
	@Test(expected = AssertionError.class)
	public void testModule70() { ... }
	```

	Without the `TODO` keyword a suppressed failure is indistinguishable from an intentional positive negative-assertion (see `testDecoratedMethod9`), and IDE task-list/`grep TODO` tooling will not surface it as temporary. The only legitimate exception is when the `AssertionError` is itself the expected positive outcome (e.g., "this function doesn't exist, so the test should fail"); those should be documented in the Javadoc as intentional.

[SO post]: https://stackoverflow.com/questions/4955635/how-to-add-local-jar-files-to-a-maven-project#answer-4955635
