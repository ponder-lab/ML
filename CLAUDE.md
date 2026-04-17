# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This is **Ariadne** — WALA-based static analysis for machine learning code in Python, with a focus on TensorFlow/Keras tensor shape and dtype inference. It is a Maven multi-module project built on [WALA](https://github.com/wala/WALA).

## Common commands

```bash
# Full build (first time, or after changes to non-test modules)
mvn clean install -DskipTests

# Run the entire test suite
mvn test

# Run a single test class
mvn -pl com.ibm.wala.cast.python.ml.test -Dtest=TestTensorflow2Model test

# Run a single test method
mvn -pl com.ibm.wala.cast.python.ml.test \
	-Dtest='TestTensorflow2Model#testDenseChain' \
	-Dsurefire.failIfNoSpecifiedTests=false test

# Run multiple specific test methods
mvn -pl com.ibm.wala.cast.python.ml.test \
	-Dtest='TestTensorflow2Model#testDenseChain+testDenseChain2' \
	-Dsurefire.failIfNoSpecifiedTests=false test

# Format code (pre-commit hook also runs these automatically)
mvn spotless:apply   # Java
black --fast .       # Python
```

Initial setup (submodules + Jython) is documented in `CONTRIBUTING.md` and is a one-time prerequisite — do not repeat it on every build.

**Test output lives in `com.ibm.wala.cast.python.ml.test/target/surefire-reports/`.** `mvn test`'s stdout is dominated by IR dumps — the `*.txt` surefire files contain the actual assertion failures. For per-method FINE logs, rerun `mvn test -Dsurefire.useFile=false` and the stdout goes to the Maven console (still very large — redirect to a file and grep).

`logging.properties` is set to FINEST at the root logger locally, so any `LOGGER.fine(...)` / `LOGGER.finer(...)` call will show up in a local run. CI uses a restricted `logging.ci.properties`. Write new diagnostic logs at `FINE`/`FINER` directly — don't add `INFO` and demote later.

## Submodules — NOT part of the codebase under change

Two Git submodules live at the repo root:

- `IDE/` — `wala/IDE` (LSP/IDE glue, unrelated to tensor analysis)
- `jython3/` — `ponder-lab/jython3.git` (Jython runtime, a large dependency)

**Do not search, grep, or glob into these directories.** They drown real results in noise. Scope searches to the non-submodule Java modules:

- `com.ibm.wala.cast.python/` — Python front-end (CAst, IR, loader)
- `com.ibm.wala.cast.python.ml/` — ML-specific analysis (tensor type inference, generators, TF/Keras modeling)
- `com.ibm.wala.cast.python.ml.test/` — JUnit tests for the ML analysis
- `com.ibm.wala.cast.python.test/` — Python test data files (`data/*.py`) and Python front-end tests

## Architecture

### The tensor type analysis pipeline

End-to-end flow for "what tensor types does a Python variable have":

```
PythonTensorAnalysisEngine.performAnalysis()
├─ builds CallGraph (PythonSSAPropagationCallGraphBuilder)
├─ getDataflowSources() — collects seed PointsToSetVariables for tensors
├─ for each source: getTensorTypes() via TensorGeneratorFactory.getGenerator()
└─ runs TensorTypeAnalysis (dataflow fixpoint over TensorVariable state)
```

- **`TensorGenerator`** (`com.ibm.wala.cast.python.ml.client`) — abstract base class representing "how to compute the shape and dtype of a tensor produced by some specific TF operation." Roughly **70+ subclasses**, one per TF op (`Ones`, `Zeros`, `Constant`, `Input`, `DenseCall`, `ElementWiseOperation`, `DatasetFromTensorSlicesGenerator`, etc.). Each overrides `getDefaultShapes(builder)` and `getDefaultDTypes(builder)`.

- **`TensorGeneratorFactory`** (same package) — dispatches from a `PointsToSetVariable` (typically the return value of a TF call) to the correct `TensorGenerator` subclass. The `getCalledFunction()` helper inspects the call site's target, including `SSABinaryOpInstruction` operands (mapped to `ADD`/`SUB`/`MUL`/`DIV` → `ElementWiseOperation`). A key internal function, `findCreator()`, walks the assignment graph backward to find the allocation site of a value.

- **`TensorTypeAnalysis`** (`com.ibm.wala.cast.python.ml.analysis`) — the WALA dataflow analysis that propagates `TensorVariable.state` (a `Set<TensorType>`) across the call graph. Seeded from `PythonTensorAnalysisEngine.getDataflowSources()`.

- **`TensorType`** (`com.ibm.wala.cast.python.ml.types`) — `(cellType: String, dims: List<Dimension<?>>)`. Dimensions are `NumericDim`, `SymbolicDim`, or `CompoundDim`. `TensorType` is **null-dims-safe**: `getDims() == null` means "unknown rank" (⊤), distinct from `getDims().isEmpty()` which means scalar rank-0.

- **TF/Keras modeling lives in `tensorflow.xml`** (in `com.ibm.wala.cast.python.ml/data/`) — this is a WALA XML summary file describing synthetic method bodies for TF APIs. Changing the XML requires `mvn clean` because of resource caching. Generators in `client/` read from the call graph built against these summaries.

### PTS vs `TensorTypeAnalysis` — two separate systems

WALA's Pointer Analysis (PA) and `TensorTypeAnalysis` serve different roles:

- The **PA** provides the assignment graph (edges between variables) and points-to sets (PTS). However, PTS for tensor variables is **often empty** — the XML summaries don't always create allocations that flow correctly through the PA to all consumers.
- **`TensorTypeAnalysis`** is a parallel type system that uses the PA's assignment graph as its flow graph but seeds tensor types independently via syntactic markers (`read_data`/`read_dataset` method names in `tensorflow.xml`). Variables with empty PTS can still have tensor types in this analysis.

When debugging why a tensor variable isn't recognized, check the `TensorTypeAnalysis` seeding path (`getDataflowSources`, `processInstruction`, `definesTensorIterable`), not just the PTS. An empty PTS doesn't mean the variable isn't a tensor — it means the seeding didn't reach it.

### Lattice conventions for `getDefaultShapes` / `getDefaultDTypes`

**Critical.** See the class-level Javadoc on `TensorGenerator` and the "Tensor Type Generators" section of `CONTRIBUTING.md`. The short version:

| Meaning | shapes | dtypes |
|---|---|---|
| Not a tensor (⊥) | `Collections.emptySet()` | `Collections.emptySet()` |
| Tensor, both unknown | `null` | `EnumSet.of(DType.UNKNOWN)` |
| Tensor, shape known, dtype unknown | `{shapes…}` | `EnumSet.of(DType.UNKNOWN)` |
| Tensor, shape unknown, dtype known | `null` | `{dtypes…}` |
| Tensor, both known | `{shapes…}` | `{dtypes…}` |

Empty set on one axis **must** be paired with empty set on the other (you can't have "not a tensor for shape but a tensor for dtype"). The most common mistake is returning `Collections.emptySet()` when you meant "unknown" (⊤) — that silently drops the variable from downstream analysis.

When adding a new `TensorGenerator` subclass, follow the checklist in `CONTRIBUTING.md > Tensor Type Generators`.

### `__call__`-body vs weight-graph codepaths

Chained layer calls (e.g., `x = self.layer1(x); x = self.layer2(x)`) are analysed through **two independent codepaths**:

1. **`__call__` body dataflow** — the IR of a user-defined `tf.keras.Model.__call__` method runs through `TensorTypeAnalysis`. `DenseCall.getDefaultShapes` reads the `inputs` argument's points-to set. When the input is the result of a prior layer call, the PTS is sometimes empty and shape inference falls back to `null`.

2. **`Model.getWeightShapes`** — walks the call graph backward from `tf.keras.Model`'s `outputs`, BFS-ing CGNode → CGNode and constructing *manual* `TensorGenerator`s per node (no PTS dependency). This path is structurally different and currently handles chained layers correctly.

`testModelAttributes*` exercises the weight-graph path. `testModelCall*` and `testModelCallConsume` exercise the `__call__`-body path. A fix for one doesn't automatically fix the other.

## Test suite

`TestTensorflow2Model` (in `com.ibm.wala.cast.python.ml.test`) is the main test class (~6800 lines, hundreds of `@Test` methods). Each test calls a `test(...)` helper with four-to-five parameters:

```java
test(
	"tf2_test_xxx.py",                           // Python test file under com.ibm.wala.cast.python.test/data/
	"functionName",                               // function to check inside the Python file
	1,                                            // expected number of tensor parameters
	1,                                            // expected number of function-local tensor variables
	Map.of(2, Set.of(TENSOR_NONE_4_FLOAT32)));   // map from parameter value number to expected types
```

The `Map` only checks **parameter** types, not all local tensor variables. To pin down a specific local tensor's type, refactor the Python file to call a sink function (`consume(x)`) on that value and assert on `consume`'s parameter.

**Tensor type constants** (`TENSOR_20_28_28_FLOAT32`, `TENSOR_NONE_4_FLOAT32`, etc.) are defined at the top of `TestTensorflow2Model`. Add new ones as needed when writing a test that expects a not-yet-represented shape.

### Python test files

Python test inputs live in `com.ibm.wala.cast.python.test/data/*.py`. When adding a new one:

1. Verify it runs to completion with `python3.10 path/to/file.py`.
2. Add `assert` statements for tensor `shape` and `dtype` at the points you expect the JUnit test to assert on — the Python runtime truth should match the static-analysis expectation.
3. Mismatches between Python asserts and JUnit expectations are a red flag either way.

### Issue-blocked tests

The `@Test(expected = AssertionError.class)` pattern (see `testModelCall5`, `testModelCallConsume`) is the idiom for "this test documents a known failure, tracked by issue #N." When adding one, always include a `TODO:` line in the Javadoc naming the blocking issue — without it, the suppression is indistinguishable from an intentional positive negative-assertion (as in `testDecoratedMethod9`) and tooling can't distinguish the two. When the underlying issue is resolved, flip the annotation to plain `@Test` and remove the `TODO` to convert the test into a positive regression guard.

## Commit style

Per `CONTRIBUTING.md`:

- End commit messages with a period.
- Use backticks for code entities: ``"Refactor `Input` definition in `tensorflow.xml`"``.

The repo uses a pre-commit hook that runs `spotless` (Java) and `black` (Python). It will sometimes reformat files after you stage them, in which case the commit is rejected with the reformatted diff sitting in the working tree — re-stage and commit again.

## Issue tracking

Issues live on `wala/ML` (the upstream repo), not on the `ponder-lab/ML` fork where development branches are pushed. When referencing issues in commit messages or Javadoc, use `wala/ML#N`. Cross-repo issue dependency features (`blocked_by`, sub-issues) are used — prefer `gh api` for linking them, since the CLI's built-in commands don't cover these.

When linking to code in an issue that references state on a feature branch, **use a commit-SHA permalink to `ponder-lab/ML`**, not a branch link on `wala/ML`/master — the code on the feature branch isn't on upstream master yet.

## Session memory

GEMINI.md at `.gemini/GEMINI.md` is an existing knowledge store that captures Gemini/Claude-learned facts about modeling decisions (e.g., why `tf.constant` needs `read_data` synthetic methods, how to handle aliasing in 1-CFA context). It's worth a read before deep modifications to `tensorflow.xml` or `TensorGenerator`/`TensorGeneratorFactory`.
