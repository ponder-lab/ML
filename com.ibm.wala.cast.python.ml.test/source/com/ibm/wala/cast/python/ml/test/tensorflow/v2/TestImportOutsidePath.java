package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.loader.ScriptOutsidePythonPathException;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.List;
import org.junit.Test;

/**
 * A script outside every PYTHONPATH entry that contains an import fails the class-hierarchy build
 * with a {@link ScriptOutsidePythonPathException} that carries the script and the path
 * (wala/ML#977) and propagates as itself, so a client decides how to deal with it.
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
    // The failure propagates as itself, so a client catches it by type and reads its fields.
    ScriptOutsidePythonPathException failure =
        assertThrows(ScriptOutsidePythonPathException.class, engine::defaultCallGraphBuilder);
    assertTrue(
        "The exception should carry the script: " + failure.getScript(),
        failure.getScript().endsWith("b.py"));
    assertEquals("The exception should carry the path.", pathFiles, failure.getPythonPath());
    String message = String.valueOf(failure.getMessage());
    assertTrue("The message should name the script: " + message, message.contains("b.py"));
    assertTrue("The message should name the path: " + message, message.contains("PYTHONPATH"));
  }
}
