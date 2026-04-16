package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.util.Util.findDefinition;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * A generator for tensors created by {@code tf.data.Dataset} transformations.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class DatasetGenerator extends TensorGenerator {

  public DatasetGenerator(PointsToSetVariable source) {
    super(source);
  }

  public DatasetGenerator(CGNode node) {
    super(node);
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
    return UNDEFINED_PARAMETER_POSITION;
  }

  @Override
  protected String getDTypeParameterName() {
    return null;
  }

  /**
   * By default, dataset transformations inherit the element shapes of the dataset they are called
   * on (the receiver). This method looks up the receiver dataset and infers shapes from it.
   *
   * @param builder the propagation call graph builder used for the analysis
   * @return a set of possible element shapes, or an empty set if unknown
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // For dataset transformations, default to shapes of the input dataset (the receiver).
    // The receiver is 'self' (arg0 in IR).
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      return this.getShapesOfValue(builder, receiverPTS);
    }
    return null;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // For dataset transformations, default to dtypes of the input dataset (the receiver).
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      return this.getDTypesOfValue(builder, receiverPTS);
    }
    return EnumSet.of(DType.UNKNOWN);
  }

  /**
   * Resolves the receiver (the dataset this one is derived from) to a {@link TensorGenerator}.
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return The generator for the receiver dataset, or {@code null} if it cannot be resolved.
   */
  public TensorGenerator getReceiverGenerator(PropagationCallGraphBuilder builder) {
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
            TensorGenerator generator = null;
            if (var != null) {
              generator = TensorGeneratorFactory.getGenerator(var, builder);
            } else {
              generator = createManualGenerator(asin.getNode(), builder);
            }
            if (generator != null) {
              return generator;
            }
          }
        }
      }
    }
    return null;
  }

  /**
   * Retrieves the sizes (number of elements) of the dataset represented by this generator. By
   * default, it recursively queries the receiver (the dataset this one is derived from).
   *
   * @param builder The propagation call graph builder used for analysis.
   * @return A set of possible dataset sizes, or an empty set if unknown.
   */
  public Set<Long> getDatasetSizes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> receiverPTS =
        this.getArgumentPointsToSet(builder, RECEIVER_PARAMETER_POSITION, SELF);
    if (receiverPTS != null && !receiverPTS.isEmpty()) {
      Set<Long> ret = HashSetFactory.make();
      for (InstanceKey valueIK : receiverPTS) {
        if (getAllocationSiteInNode(valueIK) != null) {
          AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
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
            TensorGenerator generator = null;
            if (var != null) {
              generator = TensorGeneratorFactory.getGenerator(var, builder);
            } else {
              generator = createManualGenerator(asin.getNode(), builder);
            }

            if (generator instanceof DatasetGenerator
                && !generator.getClass().equals(this.getClass())) {
              ret.addAll(((DatasetGenerator) generator).getDatasetSizes(builder));
            }
          }
        }
      }
      return ret;
    }
    return Collections.emptySet();
  }
}
