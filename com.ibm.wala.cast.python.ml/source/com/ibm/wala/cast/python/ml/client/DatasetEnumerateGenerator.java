package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.findDefinition;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.Set;

/**
 * A generator for tensors created by {@code tf.data.Dataset.enumerate}.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetEnumerateGenerator extends DatasetGenerator {

  public DatasetEnumerateGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetEnumerateGenerator(CGNode node) {
    super(node);
  }

  @Override
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    Set<TensorType> types = HashSetFactory.make();

    // Add the index type (int64 scalar)
    types.add(new TensorType(DType.INT64.name().toLowerCase(), Collections.emptyList()));

    // Add the underlying dataset types
    TensorGenerator underlying = getUnderlyingGenerator(builder);
    if (underlying != null) {
      types.addAll(underlying.getTensorTypes(builder));
    }

    return types;
  }

  /**
   * Retrieves the generator for the dataset being enumerated.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The underlying dataset generator, or null if not found.
   */
  public TensorGenerator getUnderlyingGenerator(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);

    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      for (InstanceKey valueIK : receiverPTS) {
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
        if (asin != null) {
          int vn = findDefinition(asin.getNode(), asin);
          if (vn > 0) {
            PointerKey pk =
                builder
                    .getPointerAnalysis()
                    .getHeapModel()
                    .getPointerKeyForLocal(asin.getNode(), vn);
            PointsToSetVariable var = null;
            if (!builder.getPropagationSystem().isImplicit(pk)) {
              var = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
            }
            if (var != null) {
              return TensorGeneratorFactory.getGenerator(var, builder);
            } else {
              return createManualGenerator(asin.getNode(), builder);
            }
          }
        }
      }
    }
    return null;
  }
}
