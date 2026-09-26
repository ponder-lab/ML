package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertEquals;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.ipa.callgraph.propagation.AbstractFieldPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationSystem;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.File;
import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests for wala/ML#964: a field written onto the None constant must not leak as an element of a
 * container that may be None.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TestNoneField extends AbstractTensorTest {

  /**
   * The witness: {@code stash(None, t)} writes an attribute on a receiver that is None, and {@code
   * Model.call}'s {@code pasts} may be None in the analysis, so {@code zip}'s wildcard element read
   * used to hand the stashed tensor to every block as its {@code past}; the blocks' concat then
   * carried a rankless float32 member beside the real {@code (2, 3, 4)}. With no field key on the
   * None constant the read enumerates nothing from it and the result reads the runtime shape alone.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testStashedFieldDoesNotLeak()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test("tf2_test_none_field.py", "consume", 1, 1, Map.of(2, Set.of(TENSOR_2_3_4_FLOAT32)));
  }

  /**
   * The guard: over the whole vendored NLPGNN project, no pointer key is a field of the None
   * constant. Before wala/ML#964 that project carried thirty-five such keys, eight of them
   * populated, each visible to every wildcard element read in the program.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error reading the test file.
   */
  @Test
  public void testNoFieldKeysOnNoneWholeProject()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    List<File> pathFiles = this.getPathFiles("nlpgnn_full_proj");
    PythonTensorAnalysisEngine engine =
        makeEngine(
            PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH,
            pathFiles,
            TestCorpusFixtures.NLPGNN_FULL_PROJECT_FILES);
    PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
    builder.makeCallGraph(builder.getOptions());
    PropagationSystem system = builder.getPropagationSystem();
    int noneFieldKeys = 0;
    for (Iterator<PointerKey> it = system.iteratePointerKeys(); it.hasNext(); ) {
      PointerKey pk = it.next();
      if (pk instanceof AbstractFieldPointerKey
          && PythonSSAPropagationCallGraphBuilder.isNoneConstant(
              ((AbstractFieldPointerKey) pk).getInstanceKey())) noneFieldKeys++;
    }
    assertEquals("No field key may be anchored on the None constant.", 0, noneFieldKeys);
  }
}
