package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A {@code src} layout under nested PYTHONPATH entries (wala/ML#984), on {@code src_layout}: the
 * package lives under {@code src}, and a test imports it by a dotted name ({@code import pkg.sub})
 * and reaches {@code take} through the package's initializers. The project root also holds code of
 * its own, a {@code setup.py}, which only the root's entry covers, so analyzing it puts the root on
 * the path alongside {@code src}: {@code [src_layout, src_layout/src]}. Each script was named
 * against the first entry containing it, the root, so the package was {@code src/pkg/...} and the
 * import never reached it. Each script is now named against the most specific entry containing it,
 * so the package reads alike under every path, and the root-level script is analyzed as well.
 */
public class TestNestedPythonPath extends AbstractTensorTest {

  /** The package and the test that imports it. */
  private static final String[] PACKAGE_FILES = {
    "src_layout/src/pkg/__init__.py",
    "src_layout/src/pkg/sub/__init__.py",
    "src_layout/src/pkg/sub/mod.py",
    "src_layout/tests/test_pkg.py"
  };

  /** The package and its test, plus the root-level script that puts the root on the path. */
  private static final String[] PROJECT_FILES = {
    "src_layout/setup.py",
    "src_layout/src/pkg/__init__.py",
    "src_layout/src/pkg/sub/__init__.py",
    "src_layout/src/pkg/sub/mod.py",
    "src_layout/tests/test_pkg.py"
  };

  private void testPackage(String[] files, String pythonPath)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH,
        files,
        "pkg/sub/mod.py",
        "consume",
        pythonPath,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 5))));
  }

  private void testRootScript(String pythonPath)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH,
        PROJECT_FILES,
        "setup.py",
        "consume_setup",
        pythonPath,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 2))));
  }

  /**
   * The flat path: the {@code src} and {@code tests} entries, neither inside the other. It cannot
   * cover {@code setup.py}, so the root-level script is left out.
   */
  @Test
  public void testFlatPath() throws ClassHierarchyException, CancelException, IOException {
    testPackage(PACKAGE_FILES, "src_layout/src:src_layout/tests");
  }

  /** The root first, then {@code src} inside it: on the previous engine the package is lost. */
  @Test
  public void testRootFirst() throws ClassHierarchyException, CancelException, IOException {
    testPackage(PROJECT_FILES, "src_layout:src_layout/src");
    testRootScript("src_layout:src_layout/src");
  }

  /** {@code src} first, then the root: the name does not depend on the entries' order. */
  @Test
  public void testSrcFirst() throws ClassHierarchyException, CancelException, IOException {
    testPackage(PROJECT_FILES, "src_layout/src:src_layout");
    testRootScript("src_layout/src:src_layout");
  }
}
