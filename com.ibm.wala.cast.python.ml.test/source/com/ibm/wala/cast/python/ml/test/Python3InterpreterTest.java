package com.ibm.wala.cast.python.ml.test;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assume.assumeNotNull;

import com.ibm.wala.cast.python.util.Python3Interpreter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.BeforeClass;
import org.junit.Test;

/** Unit tests for {@link Python3Interpreter}. */
public class Python3InterpreterTest {

  /**
   * Points Jython at the {@code jython3} submodule so {@link
   * org.python.core.PySystemState#initialize()} can find its bootstrap resources ({@code
   * src/resources/frozen_importlib/}). The analysis engine arranges this implicitly when it builds
   * a call graph; a standalone interpreter test must set {@code python.home} explicitly, or {@link
   * Python3Interpreter#getInterp()} fails to initialize and {@link
   * Python3Interpreter#evalAsInteger(String)} degrades to returning {@code null} for everything.
   * Locates the submodule by walking up from the working directory, resilient to running from the
   * test module or the reactor root.
   */
  @BeforeClass
  public static void pointJythonAtSubmodule() {
    for (Path c = Paths.get(System.getProperty("user.dir")); c != null; c = c.getParent()) {
      Path jython3 = c.resolve("jython3");
      if (Files.isDirectory(
          jython3.resolve("src").resolve("resources").resolve("frozen_importlib"))) {
        System.setProperty("python.home", jython3.toString());
        return;
      }
    }
  }

  /**
   * Verifies that {@link Python3Interpreter#evalAsInteger(String)} returns {@code null} rather than
   * throwing for expressions that are not constant integers, matching the {@code
   * Python2Interpreter} sibling and the method's nullable-{@code Integer} contract. Covers both
   * fallbacks: an expression that evaluates successfully to a non-integer value, and an expression
   * the embedded interpreter cannot evaluate at all (e.g., a runtime-shaped value such as {@code
   * tf.shape(x)[0]} used as a shape dimension, which previously crashed the analysis).
   */
  @Test
  public void testEvalAsIntegerReturnsNullForNonInteger() {
    Python3Interpreter interpreter = new Python3Interpreter();

    // Skip where the embedded Jython interpreter is unavailable despite the setup above (e.g.,
    // under Tycho-OSGi): the happy-path evaluation returns null there, so guarding on it avoids a
    // spurious failure.
    assumeNotNull(interpreter.evalAsInteger("3"));

    // Sanity: a constant integer evaluates to itself.
    assertEquals(Integer.valueOf(3), interpreter.evalAsInteger("3"));

    // Evaluates successfully, but to a non-integer (a float): null.
    assertNull(interpreter.evalAsInteger("2.5"));

    // Cannot be evaluated at all (an undefined name, as a runtime-shaped dimension would be): null.
    assertNull(interpreter.evalAsInteger("undefined_name_xyz[0]"));
  }
}
