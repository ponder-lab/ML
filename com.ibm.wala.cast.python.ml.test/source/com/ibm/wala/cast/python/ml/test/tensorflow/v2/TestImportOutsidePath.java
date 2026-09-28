package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import com.ibm.wala.util.WalaRuntimeException;
import java.io.File;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

/**
 * A script outside every PYTHONPATH entry that contains an import fails the class-hierarchy build
 * with a message naming the script and the path (wala/ML#977), not with a {@link
 * NullPointerException}.
 */
public class TestImportOutsidePath extends AbstractTensorTest {

  private static final String[] FILES = {
    "import_outside_path/src/pkg/__init__.py",
    "import_outside_path/src/pkg/a.py",
    "import_outside_path/outside/b.py"
  };

  @Test
  public void testImportOutsidePathNamesTheScript()
      throws ClassHierarchyException, CancelException, IOException {
    List<File> pathFiles = this.getPathFiles("import_outside_path/src");
    PythonTensorAnalysisEngine engine =
        makeEngine(PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH, pathFiles, FILES);
    // The engine wraps the class-hierarchy failure; the configuration failure is its root cause.
    Throwable failure = assertThrows(WalaRuntimeException.class, engine::defaultCallGraphBuilder);
    Throwable cause = failure;
    while (cause != null && !(cause instanceof IllegalStateException)) cause = cause.getCause();
    assertNotNull("Expected a configuration failure as the cause of " + failure + ".", cause);
    String message = String.valueOf(cause.getMessage());
    assertTrue("The message should name the script: " + message, message.contains("b.py"));
    assertTrue("The message should name the path: " + message, message.contains("PYTHONPATH"));
  }
}
