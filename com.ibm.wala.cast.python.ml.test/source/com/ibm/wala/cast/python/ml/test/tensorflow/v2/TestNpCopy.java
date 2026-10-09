package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * {@code np.copy} returns a fresh array of its argument's dtype and shape, so a copy is typed as
 * the array it copies, and a copy's rows carry that dtype through their slices and arithmetic.
 * Without a model the copy was no array the analysis knew, and the float literal beside its rows'
 * slices read beside an unresolved operand.
 *
 * <p>The arithmetic pin reads an unresolved extent where the fixture asserts {@code (4,)}, as
 * {@link TestNdarrayElements}'s do: the start-only slice {@code bc[2:]} degrades its axis
 * (wala/ML#841).
 */
public class TestNpCopy extends AbstractTensorTest {

  private static final String FILE = "tf2_test_np_copy.py";

  /**
   * The copy itself: the original's int64 dtype and its shape.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCopy() throws ClassHierarchyException, CancelException, IOException {
    test(FILE, "consume_copy", 1, 1, Map.of(2, Set.of(TensorType.of(INT_64, 2, 5))));
  }

  /**
   * The rows of a copy returned through a function that writes into it, scaled by a float literal:
   * float64, as NumPy promotes the integral row beside the literal.
   *
   * @throws ClassHierarchyException if the class hierarchy cannot be built.
   * @throws CancelException if the analysis is cancelled.
   * @throws IOException if the input fixture cannot be read.
   */
  @Test
  public void testCopiedRowArithmetic()
      throws ClassHierarchyException, CancelException, IOException {
    test(
        FILE,
        "consume_loop",
        1,
        1,
        Map.of(2, Set.of(new TensorType(FLOAT_64, List.of(UnresolvedDim.INSTANCE)))));
  }
}
