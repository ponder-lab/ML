## Gemini Added Memories
- For Python tests:
	- Verify they run to completion using `python3.10`.
	- Add `assert` statements for tensor shapes and dtypes.
	- Ensure these `assert` statements match corresponding JUnit test case expectations.
- For TensorFlow operations, always process and test positional, keyword, and mixed arguments.
- Always run `mvn clean install` first whenever a test cannot find a test file, or after modifying `tensorflow.xml`.
