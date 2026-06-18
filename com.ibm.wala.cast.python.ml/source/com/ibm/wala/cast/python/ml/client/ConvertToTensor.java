package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;
import java.util.Locale;
import java.util.Set;

/**
 * A representation of the `tf.convert_to_tensor()` API in TensorFlow.
 *
 * <p>This function converts Python objects of various types to Tensor objects. It accepts Tensor
 * objects, numpy arrays, Python lists, and Python scalars. The value-argument shape/dtype inference
 * lives in {@link ValueExtractingTensorGenerator}; this op adds a {@code dtype_hint} fallback and
 * has no explicit shape argument.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor">tf.convert_to_tensor
 *     API</a>.
 */
public class ConvertToTensor extends ValueExtractingTensorGenerator {

  protected enum Parameters {
    VALUE,
    DTYPE,
    /**
     * Optional element type for the returned tensor, used when <code>dtype</code> is <code>None
     * </code>.
     *
     * <p>Need to consider this when inferring default dtypes.
     *
     * @see <a href="https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor#dtype_hint">
     *     <code>dtype_hint</code> parameter</a>.
     */
    DTYPE_HINT,
    NAME,
    AS_REF,
    PREFERRED_DTYPE;

    public String getName() {
      return name().toLowerCase(Locale.ROOT);
    }

    public int getIndex() {
      return ordinal();
    }
  }

  public ConvertToTensor(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If the dtype argument is not specified, then the type is inferred from the type of value,
    // unless dtype_hint is provided.
    int valNum =
        this.getArgumentValueNumber(
            builder, Parameters.DTYPE_HINT.getIndex(), Parameters.DTYPE_HINT.getName(), true);
    Set<DType> defaultDTypes = super.getDefaultDTypes(builder);

    if (valNum <= 0) return defaultDTypes;

    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, Parameters.DTYPE_HINT.getIndex(), Parameters.DTYPE_HINT.getName());

    if (pointsToSet == null || pointsToSet.isEmpty()) {
      // If the argument dtype hint is not specified.
      return defaultDTypes;
    } else {
      // The dtype points-to set is non-empty, meaning that the dtype hint was explicitly set. If
      // the
      // conversion to dtype_hint is not possible, this argument has no effect. Get the dtypes from
      // the points-to set.
      Set<DType> dTypesFromDTypeHintArgument =
          this.getDTypesFromDTypeArgument(builder, pointsToSet);

      // For each possible dtype from dtype hint, check if it is compatible with default dtypes.
      Set<DType> compatibleDTypes = EnumSet.noneOf(DType.class);

      for (DType dTypeFromDTypeHint : dTypesFromDTypeHintArgument)
        for (DType defaultDType : defaultDTypes)
          if (defaultDType.canConvertTo(dTypeFromDTypeHint))
            compatibleDTypes.add(dTypeFromDTypeHint);

      // No compatible dtypes found, return the default dtypes.
      if (!compatibleDTypes.isEmpty()) return compatibleDTypes;
      else return defaultDTypes;
    }
  }
}
