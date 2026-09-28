package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertNull;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.classLoader.SourceURLModule;
import java.io.File;
import java.nio.file.Files;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * A function with a long run of basic blocks (wala/ML#981): a few thousand sequential {@code if k
 * == i:} branches. The dead call-site pass asks the block-feasibility walk about every block, and
 * the walk followed each live predecessor back to the entry, one frame per block, so on a thread
 * with a modest stack the analysis died in a {@code StackOverflowError}. The main thread's stack is
 * large enough to hide it, so the analysis runs on a thread with a 512 KB stack, as a client's
 * worker thread may. The witness is completion: on the previous engine this test ends in a {@code
 * StackOverflowError} at {@code TensorGenerator.computeBlockFeasibility}.
 */
public class TestBlockFeasibilityDepth extends AbstractTensorTest {

  /** The number of sequential branches; the previous engine overflowed at this size. */
  private static final int BRANCHES = 4000;

  /**
   * The analysis thread's requested stack size, in bytes. A {@link Thread} stack size is a request
   * the JVM may ignore; HotSpot on Linux honors it, which is what makes this test fail on the
   * previous engine. Where it is ignored, the test passes whichever engine runs.
   */
  private static final long STACK_SIZE = 512 * 1024;

  @Test
  public void testLongBranchRunCompletes() throws Exception {
    File dir = Files.createTempDirectory("block_feasibility_depth").toFile();
    StringBuilder source = new StringBuilder("import tensorflow as tf\n\n\ndef f(x, k):\n");
    for (int i = 0; i < BRANCHES; i++)
      source.append("    if k == ").append(i).append(":\n        z = k\n");
    source.append("    return x\n\n\nf(tf.ones((2, 2)), 3)\n");
    File script = new File(dir, "long_branch_run.py");
    Files.writeString(script.toPath(), source.toString());

    Throwable[] failure = new Throwable[1];
    Thread analysis =
        new Thread(
            null,
            () -> {
              try {
                PythonTensorAnalysisEngine engine =
                    new PythonTensorAnalysisEngine(
                        List.of(dir),
                        PythonTensorAnalysisEngine.TENSORFLOW,
                        PythonTensorAnalysisEngine.DEFAULT_TARGETED_CFA_DEPTH);
                engine.setModuleFiles(Set.of(new SourceURLModule(script.toURI().toURL())));
                PythonSSAPropagationCallGraphBuilder builder = engine.defaultCallGraphBuilder();
                builder.makeCallGraph(builder.getOptions());
                engine.performAnalysis(builder);
              } catch (Throwable t) {
                failure[0] = t;
              }
            },
            "block-feasibility-depth",
            STACK_SIZE);
    analysis.start();
    analysis.join();
    assertNull("The analysis failed: " + failure[0], failure[0]);
  }
}
