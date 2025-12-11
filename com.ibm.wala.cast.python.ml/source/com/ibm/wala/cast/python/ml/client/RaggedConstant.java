package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.RaggedConstant.Parameters.INNER_SHAPE;
import static com.ibm.wala.cast.python.ml.client.RaggedConstant.Parameters.RAGGED_RANK;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static java.lang.Math.max;
import static java.util.Collections.emptySet;
import static java.util.logging.Logger.getLogger;
import static java.util.stream.Collectors.toSet;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
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
import java.util.EnumSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.logging.Logger;
import java.util.stream.StreamSupport;

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

  private static Set<Integer> getPossibleInnerListLengths(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> pts) {
    Set<Integer> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey ik : pts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
      TypeReference reference = asin.getConcreteType().getReference();

      // A `list` or `tuple`.
      if (reference.equals(list) || reference.equals(tuple)) {
        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        assert objectCatalogPointsToSet.iterator().hasNext();

        InstanceKey catalogIK =
            objectCatalogPointsToSet
                .iterator()
                .next(); // Just need one element to check inner length.

        ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
        Object constantKeyValue = constantKey.getValue();

        Integer fieldIndex = (Integer) constantKeyValue;

        FieldReference subscript =
            FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

        IField f = builder.getClassHierarchy().resolveField(subscript);

        PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);

        OrdinalSet<InstanceKey> instanceFieldPointsToSet =
            pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);

        boolean containsAllListsOrTuples =
            StreamSupport.stream(instanceFieldPointsToSet.spliterator(), false)
                .allMatch(
                    ifk -> {
                      AllocationSiteInNode innerAsin = getAllocationSiteInNode(ifk);

                      if (innerAsin == null) return false;

                      TypeReference innerReference = innerAsin.getConcreteType().getReference();
                      return innerReference.equals(list) || innerReference.equals(tuple);
                    });

        if (!containsAllListsOrTuples) ret.add(objectCatalogPointsToSet.size());
        else ret.addAll(getPossibleInnerListLengths(builder, instanceFieldPointsToSet));
      } else
        throw new IllegalStateException("Expected a list or tuple, but found: " + reference + ".");
    }

    return ret;
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

  private static boolean containsScalars(PropagationCallGraphBuilder builder, InstanceKey ik) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    if (ik instanceof ConstantKey) return true; // Scalar value.
    else {
      AllocationSiteInNode asin = getAllocationSiteInNode(ik);
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
              FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

          IField f = builder.getClassHierarchy().resolveField(subscript);

          PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);

          OrdinalSet<InstanceKey> instanceFieldPointsToSet =
              pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);

          for (InstanceKey fieldIK : instanceFieldPointsToSet)
            if (containsScalars(builder, fieldIK)) return true;
        }
      } else
        throw new IllegalArgumentException(
            "Expected a list or tuple, but found: " + reference + ".");
    }

    return false;
  }

  private static int getMaximumDepthOfEmptyList(
      PropagationCallGraphBuilder builder, InstanceKey valueIK) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    int maxDepth = 0;

    AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
    TypeReference reference = asin.getConcreteType().getReference();

    // A nested `list` or `tuple`.
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
            FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

        IField f = builder.getClassHierarchy().resolveField(subscript);

        PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);

        OrdinalSet<InstanceKey> instanceFieldPointsToSet =
            pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);

        if (instanceFieldPointsToSet.isEmpty())
          // An empty list at this field.
          maxDepth = max(maxDepth, 0);

        for (InstanceKey fieldIK : instanceFieldPointsToSet) {
          int depthOfField = getMaximumDepthOfEmptyList(builder, fieldIK);
          maxDepth = max(maxDepth, 1 + depthOfField);
        }
      }
    } else
      throw new IllegalArgumentException("Expected a list or tuple, but found: " + reference + ".");

    return maxDepth;
  }

  private static int getMaximumDepthOfScalars(
      PropagationCallGraphBuilder builder, InstanceKey valueIK) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    int maxDepth = 0;

    if (valueIK instanceof ConstantKey) maxDepth = max(maxDepth, 0); // Scalar value.
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
              FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

          IField f = builder.getClassHierarchy().resolveField(subscript);

          PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);

          OrdinalSet<InstanceKey> instanceFieldPointsToSet =
              pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);

          for (InstanceKey fieldIK : instanceFieldPointsToSet) {
            int depthOfField = getMaximumDepthOfScalars(builder, fieldIK);
            maxDepth = max(maxDepth, 1 + depthOfField);
          }
        }
      } else
        throw new IllegalArgumentException(
            "Expected a list or tuple, but found: " + reference + ".");
    }

    return maxDepth;
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

    Set<InstanceKey> valuesWithScalars =
        StreamSupport.stream(valuePointsToSet.spliterator(), false)
            .filter(ik -> containsScalars(builder, ik))
            .collect(toSet());

    for (InstanceKey valueIK : valuePointsToSet) {
      int maxDepth = getMaximumDepthOfInstance(builder, valuesWithScalars, valueIK);
      LOGGER.fine("Maximum depth of `pylist`: " + maxDepth);

      // Step 2: Determine Ragged Rank (R).
      int K = maxDepth;
      LOGGER.fine("Tensor rank: " + K);

      Set<Long> rankArguments =
          this.getPossibleRaggedRankArguments(builder).stream()
              .filter(Objects::nonNull)
              .collect(toSet());

      Set<List<Dimension<?>>> innerShapeArguments =
          this.getPossibleInnerShapeArguments(builder).stream()
              .filter(Objects::nonNull)
              .collect(toSet());

      if (rankArguments.isEmpty())
        // Default ragged rank.
        if (innerShapeArguments.isEmpty()) rankArguments.add(max(0, K - 1L));
        else
          for (List<Dimension<?>> innerShape : innerShapeArguments)
            rankArguments.add(max(0, K - 1L - innerShape.size()));

      for (Long R : rankArguments) {
        LOGGER.fine("Ragged rank: " + R);

        // Step 3: Construct shape with rank K and ragged rank R.
        // The final shape is constructed by concatenating the Ragged Portion and the Uniform
        // Portion.

        // Part A: The Ragged Portion (Dimensions 0 to R)

        // For the ragged dimensions, TensorFlow does not look for a uniform length. It assigns the
        // shape based on the row_splits.

        // Get the length of the outer list.
        Set<Integer> possibleOuterListLengths =
            getPossibleOuterListLengths(builder, valuePointsToSet);

        for (int outerListLength : possibleOuterListLengths) {
          List<Dimension<?>> shape = new ArrayList<>();

          // Dim 0 (Batch): Always fixed. It is simply len(input_list).
          shape.add(new NumericDim(outerListLength));

          // The first R dimensions are ragged.
          // Dim 1 to R: These are assigned None (or ? in older outputs) in the static shape,
          // indicating they can vary.
          for (Long i = 0L; i < R; i++) shape.add(null); // Unknown size for ragged dimensions.

          // Part B: The Uniform Portion (Dimensions R + 1 to K)
          // If R < K - 1 (meaning you requested fewer ragged dimensions than the total depth),
          // TensorFlow enforces uniformity on the remaining inner dimensions.

          // 1. It checks the length of every sub-list at these levels.
          // 2. If any lengths differ, it throws a ValueError.
          // 3. If they match, that length becomes the fixed size for that dimension.

          if (R < K - 1) {
            Set<Integer> possibleInnerListLengths =
                getPossibleInnerListLengths(builder, valuePointsToSet);

            // Determine the uniform lengths for dimensions R + 1 to K - 1.
            for (long i = R + 1; i < K; i++)
              for (int innerListLength : possibleInnerListLengths)
                shape.add(new NumericDim(innerListLength));
          }

          ret.add(shape);
        }
      }
    }

    return ret;
  }

  private static int getMaximumDepthOfInstance(
      PropagationCallGraphBuilder builder,
      Set<InstanceKey> instancesWithScalars,
      InstanceKey instance) {
    if (instancesWithScalars.contains(instance)) return getMaximumDepthOfScalars(builder, instance);
    else
      // If `pylist` contains no scalar values, then K is one greater than the maximum depth of
      // empty lists in `pylist`.
      return 1 + getMaximumDepthOfEmptyList(builder, instance);
  }

  protected Set<Long> getPossibleRaggedRankArguments(PropagationCallGraphBuilder builder) {
    return this.getPossibleLongArguments(builder, this.getRaggedRankArgumentValueNumber(builder));
  }

  protected int getRaggedRankParameterPosition() {
    return RAGGED_RANK.ordinal();
  }

  protected int getRaggedRankArgumentValueNumber(PropagationCallGraphBuilder builder) {
    // TODO: Handle keyword arguments.
    return this.getArgumentValueNumber(builder, this.getRaggedRankParameterPosition(), true);
  }

  protected int getInnerShapeParameterPosition() {
    return INNER_SHAPE.ordinal();
  }

  protected int getInnerShapeArgumentValueNumber(PropagationCallGraphBuilder builder) {
    // TODO: Handle keyword arguments.
    return this.getArgumentValueNumber(builder, this.getInnerShapeParameterPosition(), true);
  }

  protected Set<List<Dimension<?>>> getPossibleInnerShapeArguments(
      PropagationCallGraphBuilder builder) {
    int valueNumber = this.getInnerShapeArgumentValueNumber(builder);

    if (valueNumber >= 0) {
      PointerKey pointerKey = builder.getPointerKeyForLocal(this.getNode(), valueNumber);
      OrdinalSet<InstanceKey> pointsToSet = builder.getPointerAnalysis().getPointsToSet(pointerKey);
      return this.getShapesFromShapeArgument(builder, pointsToSet);
    } else return emptySet();
  }

  /**
   * {@inheritDoc}
   *
   * <p>If there no scalars, we default to <code>tf.float32</code>. This isn't in the documentation,
   * but it seems to be the case.
   *
   * @see The <a href="https://github.com/tensorflow/tensorflow/issues/105858">"Update default dtype
   *     description in ragged_factory_ops.py" GitHub issue</a>.
   */
  @Override
  protected EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    int valueNumber = this.getValueArgumentValueNumber();
    PointerKey valuePK =
        pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), valueNumber);
    OrdinalSet<InstanceKey> valuePointsToSet = pointerAnalysis.getPointsToSet(valuePK);

    if (valuePointsToSet == null || valuePointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for value in source: " + this.getSource() + ".");

    if (StreamSupport.stream(valuePointsToSet.spliterator(), false)
            .filter(ik -> containsScalars(builder, ik))
            .count()
        == 0) {
      LOGGER.fine("No scalars found in `pylist`; defaulting to `tf.float32` dtype.");
      return EnumSet.of(FLOAT32);
    }

    // Otherwise, there are values available to infer the dtype from.
    return super.getDefaultDTypes(builder);
  }
}
