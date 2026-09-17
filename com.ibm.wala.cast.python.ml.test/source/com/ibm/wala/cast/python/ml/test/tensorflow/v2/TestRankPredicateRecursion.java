package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A rank assert on a tuple-unpacked op result, inside the body that produced it (wala/ML#923), on
 * {@code test_rank_assert_overflow.py}: {@code _, indices = tf.nn.top_k(scores, k=k)} followed by
 * {@code assert indices.shape.rank == 2}, with {@code k} a tensor so the op's argument read walks
 * to the callers. The caller walk filters call sites by branch reachability, which evaluates the
 * assert, whose rank predicate asks the unpacked element's generator for its shapes, whose argument
 * read walks to the callers again. With the predicate's read made directly, outside the engine's
 * memo layer, the recursion ran until the stack was exhausted and the analysis died before the
 * first type; through the memo layer the re-entrant read observes the evaluation in progress and
 * the predicate declines, so the recursion is bounded and the analysis completes. The witness is
 * completion with {@code scores} read as {@code (2, 5) float32}: on the previous engine this test
 * ends in a {@code StackOverflowError}.
 */
public class TestRankPredicateRecursion extends AbstractTensorTest {

  @Test
  public void testRankAssertOnUnpackedOpResultCompletes()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        "test_rank_assert_overflow.py",
        "pick",
        1,
        6,
        Map.of(2, Set.of(TensorType.of(FLOAT_32, 2, 5))));
  }
}
