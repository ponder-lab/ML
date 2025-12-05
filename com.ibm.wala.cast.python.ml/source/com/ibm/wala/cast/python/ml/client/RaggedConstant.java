package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static java.util.logging.Logger.getLogger;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

/**
 * A representation of the `tf.ragged.constant()` API in TensorFlow.
 *
 * @see <a
 *     href="https://www.tensorflow.org/api_docs/python/tf/ragged/constant">tf.ragged.constant</a>.
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class RaggedConstant extends ZerosLike {

  private static final Logger LOGGER = getLogger(RaggedConstant.class.getName());

  protected enum Parameters {
    PYLIST,
    DTYPE,
    RAGGED_RANK,
    INNER_SHAPE,
    NAME,
    ROW_SPLITS_DTYPE
  }

  public RaggedConstant(PointsToSetVariable source) {
    super(source);
  }

  private static Set<Integer> getPossibleOuterListLengths(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    Set<Integer> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey valueIK : valuePointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
      TypeReference reference = asin.getConcreteType().getReference();

      // A `list` or `tuple`.
      if (reference.equals(list) || reference.equals(tuple)) {
        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        ret.add(objectCatalogPointsToSet.size());
      } else
        throw new IllegalArgumentException(
            "Expected a list or tuple, but found: " + reference + ".");
    }

    return ret;
  }

  private static Set<Integer> getMaximumDepthOfScalars(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    Set<Integer> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey valueIK : valuePointsToSet) {
      int maxDepth = -1;

      if (valueIK instanceof ConstantKey) maxDepth = Math.max(maxDepth, 0); // Scalar value.
      else {
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
        TypeReference reference = asin.getConcreteType().getReference();

        // A nested `list`, `tuple`, or `np.ndarray`.
        if (reference.equals(list) || reference.equals(tuple)) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              pointerAnalysis.getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
            Object constantKeyValue = constantKey.getValue();

            Integer fieldIndex = (Integer) constantKeyValue;

            FieldReference subscript =
                FieldReference.findOrCreate(
                    Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

            IField f = builder.getClassHierarchy().resolveField(subscript);

            PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);

            OrdinalSet<InstanceKey> instanceFieldPointsToSet =
                pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);

            Set<Integer> possibleDepthsOfField =
                getMaximumDepthOfScalars(builder, instanceFieldPointsToSet);

            for (int depthOfField : possibleDepthsOfField)
              maxDepth = Math.max(maxDepth, 1 + depthOfField);
          }
        }
      }

      ret.add(maxDepth);
    }

    return ret;
  }

  @Override
  protected Set<List<Dimension<?>>> getShapesOfValue(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    // Returns a potentially ragged tensor with rank K and the specified `ragged_rank`, containing
    // the values from `pylist`.

    // All scalar values in `pylist` must have the same nesting depth K, and the returned
    // `RaggedTensor` will have rank K. If `pylist` contains no scalar values, then K is one greater
    // than the maximum depth of empty lists in `pylist`.

    // Step 1: Calculate K, the maximum depth of scalar values in `pylist`.

    if (valuePointsToSet == null || valuePointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for value in source: " + this.getSource() + ".");

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    Set<Integer> maxDepthOfScalars = getMaximumDepthOfScalars(builder, valuePointsToSet);
    LOGGER.fine("Maximum depth of scalars in `pylist`: " + maxDepthOfScalars);

    // Step 2: Determine Ragged Rank (R).
    for (int K : maxDepthOfScalars) {
      Optional<Integer> raggedRank = this.getRaggedRankArgumentValue(builder);
      int R = raggedRank.orElse(K - 1);
      LOGGER.fine("Ragged rank: " + R);

      // Step 3: Construct shape with rank K and ragged rank R.

      // Get the length of the outer list.
      Set<Integer> possibleOuterListLengths =
          getPossibleOuterListLengths(builder, valuePointsToSet);

      for (int outerListLength : possibleOuterListLengths) {
        List<Dimension<?>> shape = new ArrayList<>();
        shape.add(new NumericDim(outerListLength));

        // The first R dimensions are ragged.
        for (int i = 0; i < R; i++) shape.add(null); // Unknown size for ragged dimensions.

        /*
        // The remaining K - R dimensions are dense.
        for (int i = R; i < K; i++) {
          shape.add(new NumericDim(-1)); // Unknown size for dense dimensions.
        }
        */

        ret.add(shape);
      }
    }

    return ret;
  }

  private Optional<Integer> getRaggedRankArgumentValue(PropagationCallGraphBuilder builder) {
    // TODO Auto-generated method stub
    return Optional.empty();
  }
}
