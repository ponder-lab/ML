package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;

/**
 * A generator for {@code np.clip(a, a_min, a_max)} (<a
 * href="https://github.com/wala/ML/issues/1308">wala/ML#1308</a>). Where {@code tf.clip_by_value}
 * converts its bounds to the tensor's dtype, NumPy's clip broadcasts the array with its bounds and
 * promotes over the three, so the array's type is the result's only beside bounds that impose
 * nothing on it:
 *
 * <ul>
 *   <li>A bound that is {@code None} or a constant number is a scalar, and the result has the
 *       array's shape. Any other bound may be an array of a higher rank, and the shape is unknown.
 *   <li>A constant number takes the array's dtype where that dtype holds it, as an integer does in
 *       a floating-point array. A float beside an integral array gives {@code float64}. A number
 *       the array's dtype does not hold, as {@code -1} in a {@code uint8} array, gives a wider
 *       dtype under NumPy's value-based casting and raises from NumPy 2 on, and reads as an unknown
 *       dtype, as does any number beside an array that is not numeric.
 *   <li>Any other bound gives an unknown dtype.
 * </ul>
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.clip.html">numpy.clip</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpClip extends ClipByValue {

  /** The largest integer a 16-bit unsigned integer holds. */
  private static final long UINT16_MAX = 65535;

  /** The largest integer an 8-bit unsigned integer holds. */
  private static final long UINT8_MAX = 255;

  protected NpClip(PointsToSetVariable source) {
    super(source);
  }

  protected NpClip(CGNode node) {
    super(node);
  }

  @Override
  protected String getInputParameterName() {
    return "a";
  }

  /**
   * The values one bound of the clip may have.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param position The bound's positional index, excluding the callable.
   * @param name The bound's keyword.
   * @return The bound's constant numbers, a {@link Long} for an integer and a {@link Double} for a
   *     float, with an empty member for a bound that is {@code None}; or {@code null} when the
   *     bound may be anything else, or is not read.
   */
  private Set<Optional<Number>> getBound(
      PropagationCallGraphBuilder builder, int position, String name) {
    OrdinalSet<InstanceKey> pts = this.getArgumentPointsToSet(builder, position, name);
    // An empty set is a bound with a value the analysis cannot read. It is an omitted bound too,
    // which NumPy accepts only from 2.1 on, where the bound may instead be passed under another
    // keyword that is not read here, so neither is taken to impose nothing.
    if (pts.isEmpty()) return null;
    Set<Optional<Number>> ret = new HashSet<>();
    for (InstanceKey ik : pts) {
      if (!(ik instanceof ConstantKey)) return null;
      Object value = ((ConstantKey<?>) ik).getValue();
      if (value == null) ret.add(Optional.empty());
      // A number written in the program is a `Long` or a `Double`, and one the front end folded
      // from an expression is an `Integer` or a `Double`.
      else if (value instanceof Double) ret.add(Optional.of((Double) value));
      else if (value instanceof Long || value instanceof Integer)
        ret.add(Optional.of(((Number) value).longValue()));
      else return null;
    }
    return ret;
  }

  /**
   * The values the lower bound may have.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The values, as {@link #getBound} gives them.
   */
  private Set<Optional<Number>> getLowerBound(PropagationCallGraphBuilder builder) {
    return this.getBound(builder, 1, "a_min");
  }

  /**
   * The values the upper bound may have.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The values, as {@link #getBound} gives them.
   */
  private Set<Optional<Number>> getUpperBound(PropagationCallGraphBuilder builder) {
    return this.getBound(builder, 2, "a_max");
  }

  /**
   * The dtype NumPy gives an array of a dtype clipped by one constant bound.
   *
   * @param dtype The array's dtype.
   * @param bound The bound: a {@link Long}, a {@link Double}, or empty for none.
   * @return The result's dtype, or {@link DType#UNKNOWN} where it is not the array's own or {@code
   *     float64}.
   */
  private static DType promote(DType dtype, Optional<Number> bound) {
    if (bound.isEmpty()) return dtype;
    if (!dtype.isNumeric()) return DType.UNKNOWN;
    boolean single = dtype == DType.FLOAT32 || dtype == DType.COMPLEX64;
    if (bound.get() instanceof Double) {
      // A Python float is a double, which every integral dtype promotes to (wala/ML#814).
      if (dtype.isIntegral()) return DType.FLOAT64;
      // A magnitude that single precision does not hold gives double precision under NumPy's
      // value-based casting.
      double magnitude = Math.abs(bound.get().doubleValue());
      return single && magnitude > Float.MAX_VALUE ? DType.UNKNOWN : dtype;
    }
    long value = bound.get().longValue();
    boolean held;
    if (dtype == DType.UINT8) held = 0 <= value && value <= UINT8_MAX;
    else if (dtype == DType.INT32) held = Integer.MIN_VALUE <= value && value <= Integer.MAX_VALUE;
    // A single-precision array keeps its dtype beside an integer of at most 16 bits, and is
    // promoted to double precision beside a wider one.
    else if (single) held = Short.MIN_VALUE <= value && value <= UINT16_MAX;
    else held = true;
    return held ? dtype : DType.UNKNOWN;
  }

  /**
   * The legacy view of {@link #getDefaultShapeResult}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The result's shapes, or {@code null} when any of them is unknown.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return this.getDefaultShapeResult(builder).toLegacy();
  }

  /**
   * The array's shapes beside scalar bounds, and the unknown shape beside any other. The array's
   * own record is returned, so a partial input stays partial (wala/ML#718).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The result's shapes.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    if (this.getLowerBound(builder) == null || this.getUpperBound(builder) == null)
      return ShapeResult.unknown();
    return super.getDefaultShapeResult(builder);
  }

  /**
   * The dtypes NumPy's promotion gives the array with each pair of values its bounds may have. An
   * input that may be a Python number, list or tuple is converted as NumPy converts it, which the
   * input's read does not give, so the dtype is unknown (wala/ML#1309).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The result's dtypes, {@link DType#UNKNOWN} among them where a pair's is not known.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    Set<Optional<Number>> lowers = this.getLowerBound(builder);
    Set<Optional<Number>> uppers = this.getUpperBound(builder);
    if (lowers == null || uppers == null) return EnumSet.of(DType.UNKNOWN);
    // TODO: Read such an input as `np.array` reads it (wala/ML#1309).
    if (this.inputMayBePythonNumberOrSequence(builder)) return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (DType dtype : super.getDefaultDTypes(builder))
      for (Optional<Number> lower : lowers)
        for (Optional<Number> upper : uppers) {
          DType below = promote(dtype, lower);
          DType above = promote(dtype, upper);
          // Each is the array's dtype, `float64` beside an integral array, or unknown, so the
          // promotion over both is the one that is not the array's own, and unknown before all.
          if (below == DType.UNKNOWN || above == DType.UNKNOWN) ret.add(DType.UNKNOWN);
          else ret.add(below == dtype ? above : below);
        }
    return ret;
  }

  /**
   * The feed forwards the array's whole type only beside bounds that leave every numeric dtype as
   * it is: {@code None} and integers that the narrowest dtype holds. Beside any other constant
   * number it forwards the shape alone, and the dtype read from the array stands, and beside a
   * bound that may be an array it declares none (wala/ML#682). It declares none over an input that
   * may be a Python number, list or tuple either, whose dtype the conversion decides
   * (wala/ML#1309). A boolean array beside an integer bound, which NumPy promotes to an integer
   * dtype, is forwarded as it is, as it is through the element-wise operations.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The feed over the caller-side input keys, or {@code null} when a bound may be an array,
   *     the input may be converted, or no input key is located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    Set<Optional<Number>> lowers = this.getLowerBound(builder);
    Set<Optional<Number>> uppers = this.getUpperBound(builder);
    if (lowers == null || uppers == null) return null;
    if (this.inputMayBePythonNumberOrSequence(builder)) return null;
    Set<Optional<Number>> bounds = new HashSet<>(lowers);
    bounds.addAll(uppers);
    for (Optional<Number> bound : bounds) {
      if (bound.isEmpty()) continue;
      boolean narrow =
          bound.get() instanceof Long
              && 0 <= bound.get().longValue()
              && bound.get().longValue() <= UINT8_MAX;
      if (!narrow) return this.getTypeFeed(builder, TypeFeedKind.SHAPE_ONLY);
    }
    return super.getTypeFeed(builder);
  }

  /**
   * Returns the producing library of the modeled value: an {@code np.clip(...)} call, so the value
   * is an ndarray whatever its input is (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
