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

## Java Testing

- Always run `mvn test` to ensure all Java tests pass before committing changes.
- Add JUnit test cases for any new functionality or bug fixes implemented in the Java codebase.
- Ensure that all new and existing JUnit test cases pass successfully before committing changes.
- Use descriptive names for JUnit test methods that clearly indicate the purpose of the test.
- If you change any of the summary files (e.g., `tensorflow.xml`), ensure that you add or update JUnit test cases to cover the changes made. Also, you must run `mvn clean` to ensure that the changes are correctly reflected in the build for summary (XML) files.

[SO post]: https://stackoverflow.com/questions/4955635/how-to-add-local-jar-files-to-a-maven-project#answer-4955635
