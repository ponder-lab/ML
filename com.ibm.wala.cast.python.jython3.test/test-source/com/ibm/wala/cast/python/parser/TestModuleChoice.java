package com.ibm.wala.cast.python.parser;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import java.util.List;
import org.junit.Test;

/**
 * The script a module path names when several scripts in scope match it, each preference of the
 * ranking decided by a candidate set where it alone separates the answer (<a
 * href="https://github.com/wala/ML/issues/1042">wala/ML#1042</a>). Every set is given in both
 * orders, so the answer cannot come from the order of the scripts.
 */
public class TestModuleChoice {

  private static void assertChosen(String expected, String module, String... scripts) {
    List<String> forward = List.of(scripts);
    List<String> backward = new java.util.ArrayList<>(forward);
    java.util.Collections.reverse(backward);
    assertEquals(expected, PythonParser.chooseModule(forward, module));
    assertEquals(expected, PythonParser.chooseModule(backward, module));
  }

  /** A script named exactly by the path has no components ahead of the match, so it comes first. */
  @Test
  public void testExactName() {
    assertChosen("pkg/inputters.py", "pkg/inputters", "pkg/inputters.py", "a/pkg/inputters.py");
  }

  /**
   * The one with the fewest components ahead of the match, although the deeper one's name sorts
   * first.
   */
  @Test
  public void testFewestComponentsAhead() {
    assertChosen(
        "proj/pkg/inputters/__init__.py",
        "pkg/inputters",
        "proj/a/pkg/inputters/__init__.py",
        "proj/pkg/inputters/__init__.py");
  }

  /** At equal depth, by name. */
  @Test
  public void testNameAtEqualDepth() {
    assertChosen(
        "alpha/pkg/inputters.py",
        "pkg/inputters",
        "beta/pkg/inputters.py",
        "alpha/pkg/inputters.py");
  }

  /**
   * A module and a package of one path in one directory: the package, as Python's import system
   * takes it, although the module's name sorts first.
   */
  @Test
  public void testPackageOverModuleAtOneDepth() {
    assertChosen(
        "proj/pkg/inputters/__init__.py",
        "pkg/inputters",
        "proj/pkg/inputters/__init__.py",
        "proj/pkg/inputters.py");
  }

  /** A match only at a path-component boundary: {@code superpkg} is not {@code pkg}. */
  @Test
  public void testComponentBoundary() {
    assertChosen(
        "x/pkg/inputters.py", "pkg/inputters", "superpkg/inputters.py", "x/pkg/inputters.py");
    assertNull(PythonParser.chooseModule(List.of("superpkg/inputters.py"), "pkg/inputters"));
  }
}
