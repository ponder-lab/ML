package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests SentencePiece's id encodings: {@code SentencePieceProcessor.encode_as_ids} and {@code
 * EncodeAsIds} return a list of Python ints, so a tensor converted from one, or from a list
 * concatenation holding one, is {@code int32} of an unresolved length. Unmodeled, the encoding had
 * no elements the analysis could read, and the text generator's prompt {@code tf.expand_dims([bos]
 * + sp.encode_as_ids(text), 0)} read as an unknown dtype beside the {@code int32} draws its
 * sampling loop feeds back. A concatenation's elements sit in an order-free field, which the dtype
 * read now reads as it reads a literal's numbered elements.
 */
public class TestSentencePieceIds extends AbstractTensorTest {

  private static final String FIXTURE = "tf2_test_sentencepiece_ids.py";

  /** The model's input: the {@code (1, n)} int32 prompt, or the {@code (1, 1)} int32 draw. */
  @Test
  public void testPromptAndDraw() throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                new TensorType(INT_32, asList(new NumericDim(1), UnresolvedDim.INSTANCE)),
                TensorType.of(INT_32, 1, 1))));
  }

  /**
   * A tensor converted from {@code encode}'s result. It is not modeled: whether it returns ids or
   * pieces depends on its {@code out_type}, so the result stays unknown.
   */
  @Test
  public void testEncode() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_encode", 1, 1, Map.of(2, Set.of(TENSOR_UNKNOWN_SHAPE_UNKNOWN_DTYPE)));
  }

  /**
   * A tensor converted from {@code EncodeAsIds}'s ids: {@code int32}. Its rank is unknown, since
   * {@code tf.constant} reads no rank from a list whose elements have no known positions.
   */
  @Test
  public void testEncodeAsIdsCapitalized()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FIXTURE,
        "consume_encode_as_ids_capitalized",
        1,
        1,
        Map.of(2, Set.of(TENSOR_INT32_UNKNOWN_SHAPE)));
  }

  /** A concatenation of two int literals' lists, converted. */
  @Test
  public void testLiteralConcatenation()
      throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_literal_concat", 1, 1, Map.of(2, Set.of(TensorType.of(INT_32, 1, 3))));
  }

  /**
   * An unrelated class's {@code encode}, returning floats: the SentencePiece summary is reached
   * through a {@code SentencePieceProcessor} instance, never by the method's name.
   */
  @Test
  public void testOtherEncode() throws ClassHierarchyException, CancelException, IOException {
    test(FIXTURE, "consume_other_encode", 1, 1, Map.of(2, Set.of(TensorType.of(FLOAT_32, 2))));
  }
}
