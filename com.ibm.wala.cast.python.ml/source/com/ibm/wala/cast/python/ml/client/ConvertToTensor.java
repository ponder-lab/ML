package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.EnumSet;

/**
 * A representation of the `tf.convert_to_tensor()` API in TensorFlow.
 *
 * <p>This function converts Python objects of various types to Tensor objects. It accepts Tensor
 * objects, numpy arrays, Python lists, and Python scalars.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor">tf.convert_to_tensor
 *     API</a>.
 */
public class ConvertToTensor extends ZerosLike {

  private static final String FUNCTION_NAME = "tf.convert_to_tensor()";

  /**
   * Optional element type for the returned tensor, used when <code>dtype</code> is <code>None
   * </code>.
   *
   * <p>Need to consider this when inferring default dtypes.
   *
   * @see <a href="https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor#dtype_hint">
   *     <code>dtype_hint</code> parameter</a>.
   */
  private static final int DTYPE_HINT_PARAMETER_POSITION = 2;

  public ConvertToTensor(PointsToSetVariable source) {
    super(source);
  }

  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // If the dtype argument is not specified, then the type is inferred from the type of value,
    // unless dtype_hint is provided.
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    int valNum = this.getDTypeHintArgumentValueNumber();
    OrdinalSet<InstanceKey> pointsToSet = null;

    if (valNum > 0) {
      // The dtype hint is in an explicit argument.
      // FIXME: Handle keyword arguments.
      PointerKey pointerKey =
          pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), valNum);
      pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);
    }

    EnumSet<DType> defaultDTypes = super.getDefaultDTypes(builder);

    // If the argument dtype hint is not specified.
    if (pointsToSet == null || pointsToSet.isEmpty()) return defaultDTypes;
    else {
      // The dtype points-to set is non-empty, meaning that the dtype hint was explicitly set.
      // If the conversion to dtype_hint is not possible, this argument has no effect.

      // Get the dtypes from the points-to set.
      EnumSet<DType> dTypesFromDTypeHintArgument = getDTypesFromDTypeArgument(builder, pointsToSet);

      // for each possible dtype from dtype hint, check if it is compatible with default dtypes.
      EnumSet<DType> compatibleDTypes = EnumSet.noneOf(DType.class);

      for (DType dTypeFromDTypeHint : dTypesFromDTypeHintArgument)
        for (DType defaultDType : defaultDTypes)
          if (defaultDType.canConvertTo(dTypeFromDTypeHint))
            compatibleDTypes.add(dTypeFromDTypeHint);

      if (!compatibleDTypes.isEmpty()) return compatibleDTypes;
      else
        // No compatible dtypes found, return the default dtypes.
        return defaultDTypes;
    }
  }

  /**
   * Returns the value number for the dtype hint argument in the function call.
   *
   * @return The value number for the dtype hint argument in the function call or -1 if the dtype
   *     hint argument is not supported.
   */
  protected int getDTypeHintArgumentValueNumber() {
    return this.getArgumentValueNumber(DTYPE_HINT_PARAMETER_POSITION);
  }

  @Override
  protected String getSignature() {
    return FUNCTION_NAME;
  }
}
