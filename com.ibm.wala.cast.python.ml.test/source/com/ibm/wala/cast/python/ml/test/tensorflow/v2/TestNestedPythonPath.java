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
 * and reaches {@code take} through the package's initializers. A client that adds the project root
 * alongside {@code src} gets {@code [src_layout, src_layout/src]}; each script was named against
 * the first entry containing it, the root, so the package was {@code src/pkg/...} and the import
 * never reached it. Each script is now named against the most specific entry containing it, so the
 * three paths below read {@code take}'s callee parameter alike.
 */
public class TestNestedPythonPath extends AbstractTensorTest {

  private static final String[] FILES = {
    "src_layout/src/pkg/__init__.py",
    "src_layout/src/pkg/sub/__init__.py",
    "src_layout/src/pkg/sub/mod.py",
    "src_layout/tests/test_pkg.py"
  };

  private void testPath(String pythonPath)
      throws ClassHierarchyException, CancelException, IOException {
    test(
        PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH,
        FILES,
        "pkg/sub/mod.py",
        "consume",
        pythonPath,
        1,
        1,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 4, 5))));
  }

  /** The flat path: the {@code src} and {@code tests} entries, neither inside the other. */
  @Test
  public void testFlatPath() throws ClassHierarchyException, CancelException, IOException {
    testPath("src_layout/src:src_layout/tests");
  }

  /** The root first, then {@code src} inside it: on the previous engine the package is lost. */
  @Test
  public void testRootFirst() throws ClassHierarchyException, CancelException, IOException {
    testPath("src_layout:src_layout/src");
  }

  /** {@code src} first, then the root: the name does not depend on the entries' order. */
  @Test
  public void testSrcFirst() throws ClassHierarchyException, CancelException, IOException {
    testPath("src_layout/src:src_layout");
  }
}
