# [Ariadne](https://wala.github.io/ariadne/)

[![Continuous integration](https://github.com/ponder-lab/ML/actions/workflows/continuous-integration.yml/badge.svg)](https://github.com/ponder-lab/ML/actions/workflows/continuous-integration.yml) [![CodeQL](https://github.com/ponder-lab/ML/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ponder-lab/ML/actions/workflows/github-code-scanning/codeql) [![Automatic Dependency Submission](https://github.com/ponder-lab/ML/actions/workflows/dependency-graph/auto-submission/badge.svg)](https://github.com/ponder-lab/ML/actions/workflows/dependency-graph/auto-submission) [![Dependabot Updates](https://github.com/ponder-lab/ML/actions/workflows/dependabot/dependabot-updates/badge.svg)](https://github.com/ponder-lab/ML/actions/workflows/dependabot/dependabot-updates) [![codecov](https://codecov.io/gh/ponder-lab/ML/graph/badge.svg)](https://codecov.io/gh/ponder-lab/ML)

This is the top level repository for Ariadne code. More information on using the Ariadne tools can be found [here](https://wala.github.io/ariadne/). This repository is code to analyze machine learning code with [WALA]. Currently, the code consists of the analysis of Python (`com.ibm.wala.cast.python`), analysis focused on machine learning in Python (`com.ibm.wala.cast.python.ml`), support for using the analysis via J2EE WebSockets (`com.ibm.wala.cast.python.ml.j2ee`) and their associated test projects.

Since it is built using [WALA], you need to have WALA on your system to use it. Instructions on building this project can be found in [CONTRIBUTING.md].

To test, for example, run `TestCalls` in the `com.ibm.wala.cast.python.test` project.

## Type Annotation Sidecars

Tensor values loaded from data files (`np.load`, `pickle.load`, `tf.io.read_file`) have types the analysis cannot compute from code. A project may declare them in a sidecar file named `ariadne-types.json` at a Python-path root, or in the file named by the `ariadne.typeAnnotations.file` system property:

```json
{
"types": [
	{
	"module": "datas/loader.py",
	"function": "Planetoid.load",
	"variable": "allx",
	"dtype": "float32",
	"shape": "1708 3703",
	"attribution": "Extent of the pickled Cora feature matrix."
	}
]
}
```

`module` is the root-relative script path; `function` is the dotted qualified name within the module, omitted or empty for module level; `variable` is the source-level name of the annotated binding. `dtype` and `shape` are each optional (an absent axis stays unknown). The shape is a whitespace-separated dimension string of integers or symbolic names; `...` alone declares an unknown rank, and the empty string declares rank 0. The optional `attribution` cites the evidence that the value is content-dependent and is echoed in every report about the entry.

Annotations extend the analysis but never override it. An entry seeds a binding only where inference produced no type; a disagreement with an inferred type is reported, and the annotation is not applied. Entries that match no analyzed binding are also reported. Applied seeds carry a dedicated `ANNOTATION` provenance, keeping evidence that rests on user-supplied facts auditable downstream. See [wala/ML#370](https://github.com/wala/ML/issues/370).

[WALA]: https://github.com/wala/WALA
[CONTRIBUTING.md]: CONTRIBUTING.md#building
