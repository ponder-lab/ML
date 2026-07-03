package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;
import java.util.logging.Logger;

/**
 * Generator for the {@code __call__} on a {@code tf.keras.layers.Embedding} instance. Given an
 * integer index input of shape {@code (d1, ..., dN)}, returns a float32 tensor of shape {@code (d1,
 * ..., dN, output_dim)}, resolving {@code output_dim} from the constructor argument stored on the
 * layer instance. See <a href="https://github.com/wala/ML/issues/676">wala/ML#676</a>.
 *
 * @see <a
 *     href="https://www.tensorflow.org/versions/r2.9/api_docs/python/tf/keras/layers/Embedding">tf.keras.layers.Embedding</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class EmbeddingCall extends TensorGenerator {

  private static final Logger LOGGER = getLogger(EmbeddingCall.class.getName());

  /** The instance field carrying the constructor's {@code output_dim} argument. */
  private static final String OUTPUT_DIM_FIELD_NAME = "output_dim";

  /**
   * Constructs an {@code EmbeddingCall} from a caller-side {@link PointsToSetVariable} (the result
   * of the {@code __call__} invoke).
   *
   * @param source The {@link PointsToSetVariable} whose defining instruction is the {@code
   *     __call__} invoke on a {@code tf.keras.layers.Embedding} instance.
   */
  public EmbeddingCall(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs an {@code EmbeddingCall} anchored to a manual node.
   *
   * @param node The {@link CGNode} for the {@code __call__} synthetic method.
   */
  public EmbeddingCall(CGNode node) {
    super(node);
  }

  /**
   * Resolves the output shapes by appending the layer's {@code output_dim} to each input shape.
   * When {@code output_dim} is not statically resolvable, extending the input shapes with a dynamic
   * dimension would misstate the rank contract, so ⊤ is returned instead.
   *
   * @param builder The propagation call graph builder.
   * @return A set of output shapes, or {@code null} if the input shape or {@code output_dim} cannot
   *     be resolved.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> inputPts = this.getArgumentPointsToSet(builder, 1, "inputs");
    if (inputPts == null || inputPts.isEmpty()) return null;
    Set<List<Dimension<?>>> inputShapes = this.getShapesOfValue(builder, inputPts);
    if (inputShapes == null || inputShapes.isEmpty()) return null;

    Set<Long> outputDims = getPossibleOutputDims(builder);
    if (outputDims == null || outputDims.isEmpty()) return null;

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes)
      for (Long outputDim : outputDims) {
        if (outputDim == null) continue;
        List<Dimension<?>> outShape = new ArrayList<>(inputShape);
        outShape.add(new NumericDim(outputDim.intValue()));
        ret.add(outShape);
      }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Returns float32: embedding tables default to the layer variable dtype under the default global
   * policy, and the lookup result carries it regardless of the integer input dtype.
   *
   * @param builder The propagation call graph builder.
   * @return {@code {FLOAT32}}.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.FLOAT32);
  }

  /**
   * Resolves the {@code output_dim} values from the field the constructor summary stores on the
   * layer instance, mirroring {@code DenseCall.getPossibleUnits}.
   *
   * @param builder The propagation call graph builder.
   * @return The statically-known {@code output_dim} values, or {@code null} if the receiver or the
   *     field cannot be resolved (wala/ML#669's degrade contract applies to the values).
   */
  private Set<Long> getPossibleOutputDims(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> selfPts = this.getArgumentPointsToSet(builder, 0, "self");
    if (selfPts == null || selfPts.isEmpty()) return null;

    Set<Long> ret = HashSetFactory.make();
    for (InstanceKey selfIK : selfPts) {
      AllocationSiteInNode selfAsin = getAllocationSiteInNode(selfIK);
      if (selfAsin == null) continue;

      FieldReference outputDimRef =
          FieldReference.findOrCreate(
              selfAsin.concreteType().getReference(),
              findOrCreateAsciiAtom(OUTPUT_DIM_FIELD_NAME),
              Root);
      IField f = builder.getClassHierarchy().resolveField(outputDimRef);
      if (f == null) continue;

      PointerKey fieldPK = builder.getPointerKeyForInstanceField(selfAsin, f);
      OrdinalSet<InstanceKey> dimPts = builder.getPointerAnalysis().getPointsToSet(fieldPK);
      Set<Long> values = getPossibleLongValues(dimPts);
      LOGGER.fine(() -> "Possible `output_dim` values: " + values + " for source: " + selfIK + ".");
      if (values != null) ret.addAll(values);
    }
    return ret;
  }

  /**
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; {@code Embedding.__call__} takes no explicit
   *     shape parameter.
   */
  @Override
  protected int getShapeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; {@code Embedding.__call__} takes no explicit shape parameter.
   */
  @Override
  protected String getShapeParameterName() {
    return null;
  }

  /**
   * @return {@link #UNDEFINED_PARAMETER_POSITION}; {@code Embedding.__call__} takes no explicit
   *     dtype parameter.
   */
  @Override
  protected int getDTypeParameterPosition() {
    return UNDEFINED_PARAMETER_POSITION;
  }

  /**
   * @return {@code null}; {@code Embedding.__call__} takes no explicit dtype parameter.
   */
  @Override
  protected String getDTypeParameterName() {
    return null;
  }
}
