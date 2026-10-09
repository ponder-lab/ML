package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * Tests a self-attention layer's head split, whose input is a rank-3 {@code Dense} projection of
 * the layer's input. A wrapper returns {@code (first,) + outputs[1:]}, whose element 0 is {@code
 * first} whatever the rest holds, and an encoder loop feeds element 0 of each layer's outputs to
 * the next layer. Reading element 0 of the concatenation as any of its elements carried the rank-4
 * attention probabilities the tuple may hold after it into the next layer's input.
 */
public class TestSelfAttentionTransposeForScores extends AbstractTensorTest {

  /**
   * The head split's input is the {@code (2, 5, 8)} projection, before its own reshape. The second
   * layer's input is the first layer's context, reshaped with a {@code -1}, so its sequence axis is
   * the {@code ?} placeholder; no member is rank 4.
   */
  @Test
  public void testHeadSplitInput() throws ClassHierarchyException, CancelException, IOException {
    test(
        "tf2_test_self_attention_transpose_for_scores.py",
        "consume",
        1,
        1,
        Map.of(
            2,
            Set.of(
                TensorType.of(FLOAT_32, 2, 5, 8),
                new TensorType(
                    FLOAT_32,
                    List.of(new NumericDim(2), new SymbolicDim("?"), new NumericDim(8))))));
  }
}
