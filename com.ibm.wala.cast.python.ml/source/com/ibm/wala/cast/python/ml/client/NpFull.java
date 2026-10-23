package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.findAllocationSiteInNode;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.NumpyTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A generator for ndarrays created by NumPy's {@code full()} function.
 *
 * <p>The shape comes from the mandatory {@code shape} argument, exactly as for {@link NpOnes}. The
 * dtype comes from the {@code dtype} argument when one is given; otherwise NumPy takes it from the
 * fill value, {@code np.array(fill_value).dtype}: a Python {@code int} is {@code int64}, a {@code
 * float} {@code float64}, a {@code bool} {@code bool}, a {@code complex} {@code complex128}, and an
 * array or NumPy scalar keeps its own dtype. There is no fixed default, so a fill value whose dtype
 * cannot be read yields {@link DType#UNKNOWN}.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.full.html">numpy.full()
 *     API</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpFull extends NpOnes {

  private static final Logger LOGGER = getLogger(NpFull.class.getName());

  /** The class name of the Jython front-end's representation of a Python complex literal. */
  private static final String PYCOMPLEX_CLASS_NAME = "org.python.core.PyComplex";

  /** The parameters of {@code np.full}, in positional order. */
  protected enum Parameters {
    SHAPE,
    FILL_VALUE,
    DTYPE,
    ORDER,
    LIKE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public NpFull(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public NpFull(CGNode node) {
    super(node);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Without a {@code dtype} argument, the dtype is the fill value's NumPy dtype. A supplied
   * {@code dtype} that nothing could read degrades to {@link DType#UNKNOWN} rather than borrowing
   * the fill value's dtype (wala/ML#865).
   *
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return The fill value's NumPy dtypes, or {@link DType#UNKNOWN} when they, or a supplied {@code
   *     dtype} argument, cannot be read.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> fillValue =
        this.getArgumentPointsToSet(
            builder, Parameters.FILL_VALUE.getIndex(), Parameters.FILL_VALUE.getName());
    // A Python scalar built by the `float`, `int` or `bool` builtin has an empty points-to set;
    // NumPy converts it as a literal of its type.
    DType scalarBuiltin =
        fillValue == null || fillValue.isEmpty()
            ? this.getArgumentScalarBuiltinDType(
                builder, Parameters.FILL_VALUE.getIndex(), Parameters.FILL_VALUE.getName())
            : null;
    Set<DType> ret =
        this.dTypeApiDefaultOrUnknown(
            builder,
            scalarBuiltin != null
                ? EnumSet.of(numpyDTypeOfPythonScalar(scalarBuiltin))
                : numpyDTypesOfValue(this, builder, fillValue));

    LOGGER.fine(() -> "Default dtypes for source: " + describe(source) + " are: " + ret + ".");

    return ret;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return Parameters.DTYPE.getIndex();
  }

  @Override
  protected String getDTypeParameterName() {
    return Parameters.DTYPE.getName();
  }

  /**
   * Returns the dtypes NumPy gives an array built from the given value, {@code
   * np.array(value).dtype}, for a value that is a Python scalar, an array, or a NumPy scalar.
   * Python scalars follow NumPy's conversion ({@code int} to {@code int64}, {@code float} to {@code
   * float64}, {@code bool} to {@code bool}, {@code complex} to {@code complex128}, {@code str} to a
   * string), not TensorFlow's literal convention; an array or a NumPy scalar keeps its producer's
   * dtype. A Python sequence, whose dtype is a promotion over its elements, and anything else this
   * cannot read, yields {@link DType#UNKNOWN}.
   *
   * @param generator The generator on whose behalf a producer's dtype is read.
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @param pointsToSet The points-to set of the value.
   * @return The value's NumPy dtypes, or the singleton {@link DType#UNKNOWN}; never empty.
   */
  static Set<DType> numpyDTypesOfValue(
      TensorGenerator generator,
      PropagationCallGraphBuilder builder,
      OrdinalSet<InstanceKey> pointsToSet) {
    if (pointsToSet == null || pointsToSet.isEmpty()) return EnumSet.of(DType.UNKNOWN);

    Set<DType> ret = EnumSet.noneOf(DType.class);

    for (InstanceKey ik : pointsToSet) {
      if (ik instanceof ConstantKey) {
        Object value = ((ConstantKey<?>) ik).getValue();
        if (value instanceof Boolean) ret.add(DType.BOOL);
        else if (value instanceof Integer || value instanceof Long) ret.add(DType.INT64);
        else if (value instanceof Float || value instanceof Double) ret.add(DType.FLOAT64);
        else if (value instanceof String) ret.add(DType.STRING);
        else if (value != null && PYCOMPLEX_CLASS_NAME.equals(value.getClass().getName()))
          ret.add(DType.COMPLEX128);
        // `None` makes an object array, and any other constant has no modeled dtype.
        else return EnumSet.of(DType.UNKNOWN);
        continue;
      }

      AllocationSiteInNode asin = findAllocationSiteInNode(ik);
      if (asin == null) return EnumSet.of(DType.UNKNOWN);

      TypeReference reference = asin.concreteType().getReference();
      // A sequence's dtype is a promotion over its elements, and a dtype token is no value.
      if (reference.equals(list)
          || reference.equals(tuple)
          || reference.equals(TensorFlowTypes.D_TYPE)
          || NumpyTypes.SCALAR_TYPE_TO_DTYPE.containsKey(reference))
        return EnumSet.of(DType.UNKNOWN);

      // An array or a NumPy scalar: its producer knows its dtype.
      Set<DType> produced = generator.getDTypesFromTensor(builder, asin);
      if (produced == null || produced.isEmpty() || produced.contains(DType.UNKNOWN))
        return EnumSet.of(DType.UNKNOWN);
      ret.addAll(produced);
    }

    return ret;
  }
}
