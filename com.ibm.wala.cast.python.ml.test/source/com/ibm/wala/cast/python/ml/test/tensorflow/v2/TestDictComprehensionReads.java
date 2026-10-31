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
 * Key reads of a dict built by a comprehension over a dataset element's {@code items()} (<a
 * href="https://github.com/wala/ML/issues/1071">wala/ML#1071</a>). The read re-derives the value
 * the comprehension writes, with the value variable bound to the element's component at the key
 * read. Each read is consumed by {@code tf.expand_dims(..., -1)}, so each expected shape has a
 * trailing {@code 1}.
 */
public class TestDictComprehensionReads extends AbstractTensorTest {

  /** The fixture. */
  private static final String FIXTURE = "tf2_test_dict_comprehension_controls.py";

  /** The {@code "audios"} component sliced to one row and expanded. */
  private static final TensorType AUDIOS_SLICED = TensorType.of(FLOAT_32, 1, 3, 1);

  /** The {@code "ids"} component sliced to one row and expanded. */
  private static final TensorType IDS_SLICED = TensorType.of(INT_32, 1, 5, 1);

  /** The {@code "audios"} component expanded. */
  private static final TensorType AUDIOS = TensorType.of(FLOAT_32, 2, 3, 1);

  /**
   * The element read directly, the baseline the comprehension reads are compared with.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testDirectRead()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "direct_sink", 1, 1, Map.of(2, Set.of(AUDIOS)));
  }

  /**
   * Each key reads its own component's slice, not the join of every component's: the two keys have
   * different shapes and dtypes.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testKeyReads()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "audios_sink", 1, 1, Map.of(2, Set.of(AUDIOS_SLICED)));
    test(FIXTURE, "ids_sink", 1, 1, Map.of(2, Set.of(IDS_SLICED)));
  }

  /**
   * The same reads in the other order. A memo that ignored the key would serve the first key's
   * answer to the second, so only one of the two orders could match.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testKeyReadsReversed()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "reversed_ids_sink", 1, 1, Map.of(2, Set.of(IDS_SLICED)));
    test(FIXTURE, "reversed_audios_sink", 1, 1, Map.of(2, Set.of(AUDIOS_SLICED)));
  }

  /**
   * The identity comprehension reads the component itself.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testIdentity()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "identity_sink", 1, 1, Map.of(2, Set.of(AUDIOS)));
  }

  /**
   * An {@code if} filter only removes keys, so a key that is read was kept and reads its
   * component's value.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testFiltered()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "filtered_sink", 1, 1, Map.of(2, Set.of(AUDIOS_SLICED)));
  }

  /**
   * A value computed through a consumer that reads its operand's points-to set, which is empty for
   * a dataset element's component, still resolves: the consumer falls back to the operand's value
   * read, which the binding answers.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testPointsToConsumer()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "points_to_sink", 1, 1, Map.of(2, Set.of(AUDIOS)));
  }

  /**
   * Keyword arguments collected into a dict have keys the call names, so the comprehension over
   * them resolves.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testKeywordArguments()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "keyword_sink", 1, 1, Map.of(2, Set.of(AUDIOS_SLICED)));
  }

  /**
   * A value chosen by the key differs per key in a way the binding does not model, so the read
   * falls back to the unknown component.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testKeyDependentFallsBack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FIXTURE, "key_dependent_sink", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * A value stored into a container on its way to the written value is read back through the
   * container, so the read falls back.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testHeapEscapeFallsBack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "stacked_sink", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * A source with no known component at the key leaves nothing to bind, so the read falls back.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testUnknownKeysFallBack()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "unknown_keys_sink", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * A comprehension whose written value reads a loop-carried value that is itself the read puts the
   * re-derivation on a cycle. The result must not depend on the order the cycle is solved in, so
   * the test runs plain, with the seeds reversed, and under two cycle-order shuffles.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testCycleIsOrderInvariant()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    this.testCycle();
    String oldReverseSeeds = System.getProperty("ariadne.typeResolution.reverseSeeds");
    System.setProperty("ariadne.typeResolution.reverseSeeds", "true");
    try {
      this.testCycle();
    } finally {
      restore("ariadne.typeResolution.reverseSeeds", oldReverseSeeds);
    }
    String oldShuffleCycles =
        System.getProperty(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY);
    try {
      for (String seed : new String[] {"7", "11"}) {
        System.setProperty(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY, seed);
        this.testCycle();
      }
    } finally {
      restore(PythonTensorAnalysisEngine.SHUFFLE_CYCLES_PROPERTY, oldShuffleCycles);
    }
  }

  /**
   * Asserts the cycle's read.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  private void testCycle()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "cyclic_sink", 1, 1, Map.of(2, Set.of(AUDIOS)));
  }

  /**
   * A written value that reads another comprehension dict's key puts a comprehension read inside a
   * bound derivation; the inner read resolves under its own binding.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testNestedRead()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "nested_sink", 1, 1, Map.of(2, Set.of(AUDIOS_SLICED)));
  }

  /**
   * Two keys of one comprehension whose written value is the same read of another comprehension
   * dict reach that read under two different bindings. Neither may be cut as a recursion of the
   * other, or answered with the other's result.
   *
   * @throws ClassHierarchyException On WALA class-hierarchy error.
   * @throws IllegalArgumentException On illegal argument.
   * @throws CancelException On analysis cancellation.
   * @throws IOException On I/O error.
   */
  @Test
  public void testSharedNestedRead()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(FIXTURE, "shared_audios_sink", 1, 1, Map.of(2, Set.of(AUDIOS_SLICED)));
    test(FIXTURE, "shared_ids_sink", 1, 1, Map.of(2, Set.of(AUDIOS_SLICED)));
  }

  private static void restore(String property, String value) {
    if (value == null) System.clearProperty(property);
    else System.setProperty(property, value);
  }
}
