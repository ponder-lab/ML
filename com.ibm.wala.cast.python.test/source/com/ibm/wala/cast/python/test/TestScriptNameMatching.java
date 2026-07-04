package com.ibm.wala.cast.python.test;

import static com.ibm.wala.cast.python.loader.PythonLoader.scriptNameMatches;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/**
 * Tests {@link com.ibm.wala.cast.python.loader.PythonLoader#scriptNameMatches(String, String)}, the
 * scope-membership predicate behind the plain-import binding decision (<a
 * href="https://github.com/wala/ML/issues/687">wala/ML#687</a>). Depending on how the embedding
 * client constructs its modules, collected entry names may be bare, project-relative, or the
 * absolute filesystem path of the checkout, while the importer always looks up the script-relative
 * name.
 */
public class TestScriptNameMatching {

  /** A bare collected name matches itself. */
  @Test
  public void testExactMatch() {
    assertTrue(scriptNameMatches("B.py", "B.py"));
  }

  /** A project-relative collected name matches the bare lookup. */
  @Test
  public void testProjectRelativeMatch() {
    assertTrue(scriptNameMatches("proj/B.py", "B.py"));
  }

  /**
   * An absolute collected name matches the bare lookup — the wala/ML#687 failing-desktop shape,
   * where every entry is collected under the checkout's absolute path.
   */
  @Test
  public void testAbsolutePathMatch() {
    assertTrue(
        scriptNameMatches(
            "/home/user/git/Client/tests/resources/Function/testModule6/in/B.py", "B.py"));
  }

  /** A dotted (package-qualified) lookup matches at the same suffix boundary. */
  @Test
  public void testPackageQualifiedLookupMatch() {
    assertTrue(scriptNameMatches("/home/user/proj/custom/layers.py", "custom/layers.py"));
  }

  /** The suffix must start at a path-component boundary. */
  @Test
  public void testNoPartialComponentMatch() {
    assertFalse(scriptNameMatches("/home/user/proj/AB.py", "B.py"));
  }

  /** A different file name does not match. */
  @Test
  public void testDifferentNameNoMatch() {
    assertFalse(scriptNameMatches("/home/user/proj/C.py", "B.py"));
  }

  /** The lookup being longer than the collected name does not match. */
  @Test
  public void testLongerLookupNoMatch() {
    assertFalse(scriptNameMatches("B.py", "in/B.py"));
  }
}
