package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.Loggables.describe;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for a float-valued attribute of {@code np.finfo(dtype)}: {@code max}, {@code min},
 * {@code eps}, {@code tiny} and their siblings, each a NumPy scalar of the queried type, so a
 * rank-0 array whose dtype is the {@code dtype} argument's (wala/ML#907).
 *
 * <p>The attribute's value is never modeled, only its shape and dtype: what a program does with a
 * machine limit is scale a tensor by it ({@code labels * MAX_FLOAT} with {@code MAX_FLOAT =
 * np.finfo(np.float32).max / 100.0}), and a broadcast against a rank-0 operand keeps the tensor's
 * shape. Before the attribute was modeled the operand was opaque and not provably scalar, so the
 * product floored to "not a tensor" and every consumer downstream, {@code tf.nn.top_k} included,
 * lost the rank.
 *
 * <p>The generator anchors at the helper call {@code finfo.do} makes for the attribute value (the
 * wala/ML#834 helper-call pattern), whose frame mirrors {@code finfo}'s own {@code self dtype}
 * layout; an unresolvable {@code dtype} keeps the rank-0 shape, since scalarness does not depend on
 * the type, and degrades the dtype to unknown.
 *
 * <p><b>The dtype is the union over the program's {@code np.finfo} sites.</b> The helper node is
 * one node shared by every {@code finfo} call context, and a value read off an attribute is typed
 * by delegation to the helper's allocation, so the {@code dtype} argument it reads is the union of
 * every site's. A program with one site, or sites of one type, reads exactly; a program querying
 * several types reads all of them at each site. That imprecision grows with how often the program
 * uses the API rather than with the construct, which is unusual and worth knowing. The union is
 * sound (the true dtype is a member), a disagreeing dtype set declines downstream rather than
 * picking a member, and the shape half is unaffected. Anchoring the read at the {@code finfo} frame
 * would not help: the delegated read starts from the shared allocation and cannot tell which frame
 * it was reached from. See {@code TestConstructors#testNumpyFinfoDTypes}.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.finfo.html">numpy.finfo</a>
 * @see NpScalarType
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpFinfoValue extends TensorGenerator {

  private static final Logger LOGGER = getLogger(NpFinfoValue.class.getName());

  /** The positional index of the queried type, {@code self} excluded. */
  private static final int DTYPE_PARAMETER_POSITION = 0;

  /** The keyword name of the queried type. */
  private static final String DTYPE_PARAMETER_NAME = "dtype";

  /**
   * Constructs a generator anchored to the helper call's result.
   *
   * @param source The {@link PointsToSetVariable} the helper call defines.
   */
  public NpFinfoValue(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs a generator anchored to the allocating synthetic node, for producer delegation.
   *
   * @param node The {@link CGNode} for the helper's {@code do()} method.
   */
  public NpFinfoValue(CGNode node) {
    super(node);
  }

  /**
   * A machine limit is a scalar whatever the queried type, so the shape is rank 0 unconditionally.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The singleton rank-0 shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Collections.singleton(Collections.emptyList());
  }

  /**
   * The dtype of the queried type: {@code np.finfo(np.float32).max} is a {@code float32} scalar,
   * {@code np.finfo(float).eps} a {@code float64} one. A {@code dtype} the analysis cannot read
   * leaves the dtype unknown rather than guessed.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The queried type's dtype, or {@link DType#UNKNOWN} when it does not resolve.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> dtypePts =
        this.getArgumentPointsToSet(builder, DTYPE_PARAMETER_POSITION, DTYPE_PARAMETER_NAME);
    if (dtypePts == null || dtypePts.isEmpty()) {
      LOGGER.fine(
          () ->
              "finfo dtype argument unresolved for "
                  + describe(this.getSource())
                  + "; dtype unknown.");
      return EnumSet.of(DType.UNKNOWN);
    }
    Set<DType> dtypes;
    try {
      dtypes = this.getDTypesFromDTypeArgument(builder, dtypePts);
    } catch (IllegalArgumentException e) {
      dtypes = null;
    }
    return dtypes == null || dtypes.isEmpty() ? EnumSet.of(DType.UNKNOWN) : dtypes;
  }

  /**
   * Returns the producing library of the modeled value: a NumPy machine limit is a NumPy scalar
   * (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }

  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getShapeParameterName() {
    return null;
  }

  @Override
  protected int getDTypeParameterPosition() {
    return DTYPE_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return DTYPE_PARAMETER_NAME;
  }
}
