package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static org.junit.Assert.assertTrue;

import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine;
import com.ibm.wala.cast.python.ml.client.TensorGeneratorFactory;
import com.ibm.wala.classLoader.SourceURLModule;
import java.io.File;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import org.junit.Test;

/**
 * A long straight-line chain of binary operators over a tensor. Deciding whether a binary operator
 * has a tensor operand asks for the operand's generator, the operand is the previous step's sum,
 * and nothing bounds the recursion, so the generator dispatch descends one round of frames per step
 * and overflows the stack. The analysis runs on a thread with a 512 KB stack, as in {@link
 * TestBlockFeasibilityDepth}, which keeps the chain short enough to overflow quickly.
 *
 * <p>Where the overflow begins is not fixed: measured on one machine, the same stack overflowed at
 * 500 steps in one JVM and not in another, and at 600 steps and beyond in every run, at 512 KB and
 * at 2 MB. So the chain is twice the length that failed to overflow. A longer one would add margin
 * but cost time: at 2,000 steps the analysis takes about 22 seconds against about 5 at 1,000. The
 * stack cannot shrink instead, since at 128 KB the thread overflows loading classes before the
 * analysis starts, which the check on the overflow's frames rejects.
 *
 * <p>TODO: Expect the analysis to complete once <a
 * href="https://github.com/wala/ML/issues/982">wala/ML#982</a> is fixed.
 */
public class TestBinopChainDepth extends AbstractTensorTest {

  /** The number of steps in the chain. */
  private static final int STEPS = 1000;

  /**
   * The analysis thread's requested stack size, in bytes. A {@link Thread} stack size is a request
   * the JVM may ignore; HotSpot on Linux honors it.
   */
  private static final long STACK_SIZE = 512 * 1024;

  @Test
  public void testLongBinopChainOverflows() throws Exception {
    File dir = Files.createTempDirectory("binop_chain_depth").toFile();
    StringBuilder source =
        new StringBuilder(
            "import tensorflow as tf\n\n\ndef outer(x):\n    total = tf.reduce_sum(x)\n");
    for (int i = 0; i < STEPS; i++)
      source
          .append("    f")
          .append(i)
          .append(" = lambda: x * ")
          .append(i)
          .append(".0\n    total = total + tf.reduce_sum(f")
          .append(i)
          .append("())\n");
    source.append("    return total\n\n\nouter(tf.constant([1.0, 2.0]))\n");
    File script = new File(dir, "binop_chain.py");
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
            "binop-chain-depth",
            STACK_SIZE);
    analysis.start();
    analysis.join();
    assertTrue(
        "Expected a StackOverflowError, but the analysis ended with: " + failure[0],
        failure[0] instanceof StackOverflowError);
    assertTrue(
        "The stack overflowed outside the generator dispatch.",
        Arrays.stream(failure[0].getStackTrace())
            .anyMatch(
                frame -> frame.getClassName().equals(TensorGeneratorFactory.class.getName())));
  }
}
