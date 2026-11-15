package com.ibm.wala.cast.python.ml.test.tensorflow.v2;

import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.INT_64;
import static com.ibm.wala.cast.python.ml.test.tensorflow.v2.AbstractTensorTest.UNKNOWN;
import static java.util.Arrays.asList;

import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.cha.ClassHierarchyException;
import com.ibm.wala.util.CancelException;
import java.io.IOException;
import java.util.Map;
import java.util.Set;
import org.junit.Test;

/**
 * A value typed only by the project's {@code ariadne-types.json} sidecar, an unpickled int64
 * sequence named {@code data}, appended to with {@code np.append} in three forms, on {@code
 * sidecar_append_loop_proj} (<a href="https://github.com/wala/ML/issues/1264">wala/ML#1264</a>).
 * Each form's result is a rank-1 int64 array at run time.
 */
public class TestSidecarAppendLoop extends AbstractTensorTest {

  private static final String[] FILES = {"sidecar_append_loop_proj/append_loop.py"};

  private static final String PROJECT = "sidecar_append_loop_proj";

  private static final TensorType INT64_UNRESOLVED =
      new TensorType(INT_64, asList(UnresolvedDim.INSTANCE));

  /**
   * One {@code np.append}, then another in a loop: {@code int64} of unresolved extent, beside an
   * unknown-dtype member. The caller seeds the call result with the join of the callee's two {@code
   * np.append} creators, whose generators cannot read the dtype the sidecar supplies only through
   * the dataflow, and the annotation's refinement of such a seed covers one binding's objects, not
   * the two the loop's result may hold.
   *
   * <p>TODO: https://github.com/wala/ML/issues/1264; drop the unknown member once the caller's seed
   * no longer reads a creator whose dtype only the dataflow types.
   */
  @Test
  public void testAppendThenLoop()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "append_loop.py",
        "consume_append_then_loop",
        PROJECT,
        1,
        1,
        Map.of(
            2, Set.of(INT64_UNRESOLVED, new TensorType(UNKNOWN, asList(UnresolvedDim.INSTANCE)))));
  }

  /**
   * Two {@code np.append}s in a row: {@code int64} alone, since the caller's seed holds only the
   * second append's result, which the annotation describes.
   */
  @Test
  public void testAppendTwice()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "append_loop.py",
        "consume_append_twice",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(INT64_UNRESOLVED)));
  }

  /**
   * A loop of {@code np.append} with no append before it: {@code int64} alone, the loaded value's
   * annotated {@code (n,)} when the loop does not run and of unresolved extent when it does.
   */
  @Test
  public void testLoopOnly()
      throws ClassHierarchyException, IllegalArgumentException, CancelException, IOException {
    test(
        FILES,
        "append_loop.py",
        "consume_loop_only",
        PROJECT,
        1,
        1,
        Map.of(2, Set.of(new TensorType(INT_64, asList(new SymbolicDim("n"))), INT64_UNRESOLVED)));
  }
}
