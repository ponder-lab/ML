package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests that a subclass of {@code tf.keras.layers.Embedding} gathering from its own weight reads
 * the weight's {@code (input_dim, output_dim)} float32 type: a rank-3 gather over the rank-2
 * weight, where the weight was untyped and the gather rankless.
 *
 * <p>Each pin also holds a member with unresolved extents: the subclass's synthesized constructor
 * dispatches {@code __init__} to the inherited one as well as to the subclass's own, and on that
 * path the constructor's arguments do not reach the inherited one's dimensions. The call through
 * {@code super().__init__(*args, **kwargs)} binds them, and gives the exact member.
 */
public class TestEmbeddingSubclassWeight extends AbstractTensorTest {

  private static final String FILE = "tf2_test_embedding_subclass_weight.py";

  /**
   * The subclass's gather from {@code self.embeddings}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testGathered() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_gathered",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 1, 3, 8),
                new TensorType(
                    FLOAT_32,
                    List.of(new NumericDim(1), new NumericDim(3), UnresolvedDim.INSTANCE)))));
  }

  /**
   * The subclass's result beside a plain {@code Embedding}'s.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testSum() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_sum",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 1, 3, 8),
                new TensorType(
                    FLOAT_32,
                    List.of(new NumericDim(1), new NumericDim(3), UnresolvedDim.INSTANCE)))));
  }

  /**
   * The weight itself, {@code (input_dim, output_dim)} float32, read off the instance: a pin the
   * layer's own call rule cannot satisfy, and one that sees {@code input_dim}, which the gather
   * drops.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testWeight() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_weight",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 11, 8),
                new TensorType(
                    FLOAT_32, List.of(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)))));
  }

  /**
   * A second instance, its {@code output_dim} passed by keyword through the subclass's {@code
   * **kwargs}: its own {@code (5, 4)} weight, and nothing of the first instance's dimensions.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testKeywordWeight() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_keyword_weight",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 5, 4),
                new TensorType(
                    FLOAT_32, List.of(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)))));
  }

  /**
   * An instance whose {@code output_dim} the analysis cannot read: the column count is unknown, not
   * {@code input_dim}. An identity matrix's column count defaults to its row count, an embedding
   * weight's does not, so the weight is {@code (11, ?)} and never the square {@code (11, 11)}.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testUnreadWeight() throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_unread_weight",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(FLOAT_32, List.of(new NumericDim(11), UnresolvedDim.INSTANCE)),
                new TensorType(
                    FLOAT_32, List.of(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE)))));
  }
}
