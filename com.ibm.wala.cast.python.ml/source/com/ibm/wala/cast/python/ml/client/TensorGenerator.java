package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.D_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSORFLOW;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.ContextItem;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.types.Descriptor;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.logging.Logger;

public abstract class TensorGenerator {

  protected static final Logger LOGGER = Logger.getLogger(TensorGenerator.class.getName());

  private static final MethodReference IMPORT =
      MethodReference.findOrCreate(
          TENSORFLOW,
          findOrCreateAsciiAtom("import"),
          Descriptor.findOrCreate(null, TENSORFLOW.getName()));

  protected PointsToSetVariable source;

  protected CGNode node;

  public TensorGenerator(PointsToSetVariable source, CGNode node) {
    this.source = source;
    this.node = node;
  }

  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> shapes = getShapes(builder);
    EnumSet<DType> dTypes = getDTypes(builder);

    Set<TensorType> ret = HashSetFactory.make();

    // Create a tensor type for each possible shape and dtype combination.
    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));

    return ret;
  }

  /**
   * Returns the possible shapes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param pointsToSet The points-to set of the shape argument.
   * @return A set of possible shapes of the tensor returned by this generator.
   */
  protected Set<List<Dimension<?>>> getShapesFromShapeArgument(
      PropagationCallGraphBuilder builder, Iterable<InstanceKey> pointsToSet) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey instanceKey : pointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
      TypeReference reference = asin.getConcreteType().getReference();

      if (reference.equals(list)) { // TODO: This can also be a tuple of tensors.
        // We have a list of integers that represent the shape.
        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        // We expect the object catalog to contain a list of integers. Each element in the array
        // corresponds to the set of possible dimensions for that index.
        @SuppressWarnings("unchecked")
        Set<Dimension<Integer>>[] possibleDimensions = new Set[objectCatalogPointsToSet.size()];

        for (InstanceKey catalogIK : objectCatalogPointsToSet) {
          ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
          Object constantKeyValue = constantKey.getValue();

          Integer fieldIndex = (Integer) constantKeyValue;

          FieldReference subscript =
              FieldReference.findOrCreate(
                  PythonTypes.Root, findOrCreateAsciiAtom(fieldIndex.toString()), PythonTypes.Root);

          IField f = builder.getClassHierarchy().resolveField(subscript);
          LOGGER.fine("Found field: " + f);

          // We can now get the pointer key for the instance field.
          PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);
          LOGGER.fine("Found pointer key for instance field: " + pointerKeyForInstanceField + ".");

          // Get the points-to set for the instance field.
          OrdinalSet<InstanceKey> instanceFieldPointsToSet =
              pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);
          LOGGER.fine("Points-to set for instance field: " + instanceFieldPointsToSet + ".");

          // If the instance field points to a constant, we can use it as the shape.
          // TODO: Is it possible to also do it for (simple) expressions?
          Set<Dimension<Integer>> tensorDimensions = HashSetFactory.make();

          for (InstanceKey instanceFieldIK : instanceFieldPointsToSet) {
            if (instanceFieldIK instanceof ConstantKey) {
              // We have a constant key.
              ConstantKey<?> instanceFieldConstant = (ConstantKey<?>) instanceFieldIK;
              Object instanceFieldValue = instanceFieldConstant.getValue();

              // We have a shape value.
              Long shapeValue = (Long) instanceFieldValue;
              LOGGER.fine(
                  "Found shape value: " + shapeValue + " for " + source.getPointerKey() + ".");

              Dimension<Integer> dimension = new NumericDim(shapeValue.intValue());

              LOGGER.fine("Adding dimension: " + dimension + ".");
              tensorDimensions.add(dimension);
            } else
              throw new IllegalStateException(
                  "Expected a constant key for instance field: "
                      + pointerKeyForInstanceField
                      + ", but got: "
                      + instanceFieldIK
                      + ".");
          }

          LOGGER.info(
              "Found possible shape dimensions: "
                  + tensorDimensions
                  + " for field: "
                  + pointerKeyForInstanceField
                  + " for source: "
                  + source
                  + ".");

          // Add the shape dimensions.
          assert possibleDimensions[fieldIndex] == null
              : "Duplicate field index: "
                  + fieldIndex
                  + " in object catalog: "
                  + objectCatalogPointsToSet
                  + ".";

          possibleDimensions[fieldIndex] = tensorDimensions;
          LOGGER.fine(
              "Added shape dimensions: "
                  + tensorDimensions
                  + " for field index: "
                  + fieldIndex
                  + ".");
        }

        for (int i = 0; i < possibleDimensions.length; i++)
          for (Dimension<Integer> iDim : possibleDimensions[i]) {
            @SuppressWarnings("unchecked")
            Dimension<Integer>[] dimensions = new Dimension[possibleDimensions.length];

            dimensions[i] = iDim;

            for (int j = 0; j < possibleDimensions.length; j++)
              if (i != j)
                for (Dimension<Integer> jDim : possibleDimensions[j]) dimensions[j] = jDim;

            ret.add(asList(dimensions));
          }
      } else
        throw new IllegalStateException(
            "Expected a " + PythonTypes.list + " for the shape, but got: " + reference + ".");
    }

    return ret;
  }

  protected abstract Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder);

  protected abstract int getValueNumberForShapeArgument();

  /**
   * Returns the possible shapes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return a set of shapes, where each shape is represented as a list of dimensions
   */
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    // Get the shape from the explicit argument.
    // FIXME: Handle keyword arguments.
    int shapeArgValueNum = this.getValueNumberForShapeArgument();

    PointerKey pointerKey =
        pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, shapeArgValueNum);
    OrdinalSet<InstanceKey> pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);

    // If the argument shape is not specified.
    if (pointsToSet.isEmpty()) return getDefaultShapes(builder);
    else
      // The shape points-to set is non-empty, meaning that the shape was explicitly set.
      return getShapesFromShapeArgument(builder, pointsToSet);
  }

  /**
   * Returns the possible shapes of the tensor returned by this generator. The shape is inferred
   * from the argument represented by the given value number.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param valueNumber The value number of the argument from which to infer the shape.
   * @return A set of possible shapes of the tensor returned by this generator.
   */
  protected Set<List<Dimension<?>>> getShapes(
      PropagationCallGraphBuilder builder, int valueNumber) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    PointerKey valuePK = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, valueNumber);
    OrdinalSet<InstanceKey> valuePointsToSet = pointerAnalysis.getPointsToSet(valuePK);
    return getShapesOfValue(builder, valuePointsToSet);
  }

  /**
   * Returns the possible shapes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param pointsToSet The points-to set of the value from which the shape will be derived.
   * @return A set of possible shapes of the tensor returned by this generator.
   */
  private Set<List<Dimension<?>>> getShapesOfValue(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey valueIK : valuePointsToSet)
      if (valueIK instanceof ConstantKey) ret.add(emptyList()); // Scalar value.
      else if (valueIK instanceof AllocationSiteInNode) {
        AllocationSiteInNode asin = (AllocationSiteInNode) valueIK;
        TypeReference reference = asin.getConcreteType().getReference();

        if (reference.equals(list)) {
          OrdinalSet<InstanceKey> objectCatalogPointsToSet =
              pointerAnalysis.getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));

          LOGGER.fine(
              "The object catalog points-to set size is: " + objectCatalogPointsToSet.size() + ".");

          for (InstanceKey catalogIK : objectCatalogPointsToSet) {
            ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
            Object constantKeyValue = constantKey.getValue();

            Integer fieldIndex = (Integer) constantKeyValue;

            FieldReference subscript =
                FieldReference.findOrCreate(
                    PythonTypes.Root,
                    findOrCreateAsciiAtom(fieldIndex.toString()),
                    PythonTypes.Root);

            IField f = builder.getClassHierarchy().resolveField(subscript);
            LOGGER.fine("Found field: " + f);

            PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);
            LOGGER.fine(
                "Found pointer key for instance field: " + pointerKeyForInstanceField + ".");

            OrdinalSet<InstanceKey> instanceFieldPointsToSet =
                pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);
            LOGGER.fine("Points-to set for instance field: " + instanceFieldPointsToSet + ".");

            Set<List<Dimension<?>>> shapesOfField =
                getShapesOfValue(builder, instanceFieldPointsToSet);

            for (List<Dimension<?>> shapeList : shapesOfField) {
              List<Dimension<?>> shape = new ArrayList<>();

              shape.add(new NumericDim(objectCatalogPointsToSet.size()));
              shape.addAll(shapeList);

              ret.add(shape);
            }
          }
        } else throw new IllegalStateException("Unknown type reference: " + reference + ".");
      } else
        throw new IllegalStateException(
            "Expected a " + ConstantKey.class + " for value, but got: " + valueIK + ".");

    return ret;
  }

  /**
   * Returns the possible dtypes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param pointsToSet The points-to set of the dtype argument, which is expected to be a set of
   *     type literals.
   * @return A set of possible dtypes of the tensor returned by this generator.
   */
  protected EnumSet<DType> getDTypes(
      PropagationCallGraphBuilder builder, Iterable<InstanceKey> pointsToSet) {
    EnumSet<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey instanceKey : pointsToSet) {
      IClass concreteType = instanceKey.getConcreteType();
      TypeReference typeReference = concreteType.getReference();

      if (typeReference.equals(TensorFlowTypes.D_TYPE)) {
        // we have a dtype.
        // let's see if it's float32.
        Set<CGNode> importNodes = builder.getCallGraph().getNodes(IMPORT);

        // find the import node from this file.
        Optional<CGNode> importNode =
            importNodes.stream()
                .filter(
                    in -> {
                      ContextItem contextItem = in.getContext().get(CALL_STRING);
                      CallString cs = (CallString) contextItem;

                      // We expect the first method in the call string to be the import.
                      assert cs.getMethods().length == 1
                          : "Expected a single method in the call string, but got: "
                              + cs.getMethods().length
                              + " for node: "
                              + in;

                      IMethod method = cs.getMethods()[0];

                      CallString nodeCS = (CallString) node.getContext().get(CALL_STRING);

                      // We expect the first method in the call string to be the import.
                      assert nodeCS.getMethods().length == 1
                          : "Expected a single method in the call string, but got: "
                              + nodeCS.getMethods().length
                              + " for node: "
                              + in;

                      return method.equals(nodeCS.getMethods()[0]);
                    })
                .findFirst();

        InstanceKey tensorFlowIK =
            pointerAnalysis
                .getHeapModel()
                .getInstanceKeyForAllocation(
                    importNode.get(), NewSiteReference.make(0, TENSORFLOW));

        FieldReference float32 =
            FieldReference.findOrCreate(
                PythonTypes.Root, findOrCreateAsciiAtom(FLOAT32.name().toLowerCase()), D_TYPE);

        IField float32Field = builder.getClassHierarchy().resolveField(float32);

        PointerKey float32PK =
            pointerAnalysis
                .getHeapModel()
                .getPointerKeyForInstanceField(tensorFlowIK, float32Field);

        for (InstanceKey float32IK : pointerAnalysis.getPointsToSet(float32PK))
          if (float32IK.equals(instanceKey)) {
            ret.add(FLOAT32);
            LOGGER.info(
                "Found dtype: "
                    + FLOAT32
                    + " for source: "
                    + source
                    + " from dType: "
                    + instanceKey
                    + ".");
          } else throw new IllegalStateException("Unknown dtype: " + instanceKey + ".");
      } else
        throw new IllegalStateException(
            "Expected a "
                + TensorFlowTypes.D_TYPE
                + " for the dtype, but got: "
                + typeReference
                + ".");
    }

    return ret;
  }

  /**
   * Returns a set of possible dtypes of the tensor returned by this generator when an explicit
   * dtype isn't provided as an argument.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible dtypes of the tensor returned by this generator when an explicit
   *     dtype isn't provided as an argument.
   */
  protected abstract EnumSet<DType> getDefaultDTypes(PropagationCallGraphBuilder builder);

  /**
   * Returns the value number for the dtype argument in the function call.
   *
   * @return The value number for the dtype argument in the function call or -1 if the dtype
   *     argument is not supported.
   */
  protected abstract int getValueNumberForDTypeArgument();

  protected EnumSet<DType> getDTypes(PropagationCallGraphBuilder builder) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    int valNum = this.getValueNumberForDTypeArgument();
    OrdinalSet<InstanceKey> pointsToSet = null;

    if (valNum > 0) {
      // The dtype is in an explicit argument.
      // FIXME: Handle keyword arguments.
      PointerKey pointerKey = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, valNum);
      pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);
    }

    // If the argument dtype is not specified.
    if (pointsToSet == null || pointsToSet.isEmpty()) return getDefaultDTypes(builder);
    else
      // The dtype points-to set is non-empty, meaning that the dtype was explicitly set.
      return getDTypes(builder, pointsToSet);
  }

  /**
   * Returns the possible dtypes of the tensor returned by this generator. The dtype is inferred
   * from the argument represented by the given value number.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param valueNumber The value number of the argument from which to infer the dtype.
   * @return A set of possible dtypes of the tensor returned by this generator.
   */
  protected EnumSet<DType> getDTypes(PropagationCallGraphBuilder builder, int valueNumber) {
    EnumSet<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    PointerKey valuePK = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, valueNumber);

    for (InstanceKey valueIK : pointerAnalysis.getPointsToSet(valuePK))
      if (valueIK instanceof ConstantKey) { // It's a scalar value.
        ConstantKey<?> constantKey = (ConstantKey<?>) valueIK;
        Object value = constantKey.getValue();
        if (value instanceof Float || value instanceof Double) {
          ret.add(FLOAT32);
          LOGGER.info(
              "Inferred dtype: "
                  + FLOAT32
                  + " for source: "
                  + source
                  + " from value: "
                  + value
                  + ".");
        } else if (value instanceof Integer || value instanceof Long) {
          ret.add(INT32);
          LOGGER.info(
              "Inferred dtype: "
                  + INT32
                  + " for source: "
                  + source
                  + " from value: "
                  + value
                  + ".");
        } else if (value instanceof String) {
          ret.add(STRING);
          LOGGER.info(
              "Inferred dtype: "
                  + STRING
                  + " for source: "
                  + source
                  + " from value: "
                  + value
                  + ".");
        } else throw new IllegalStateException("Unknown constant type: " + value.getClass() + ".");
      } else
        // TODO: More cases.
        throw new IllegalStateException(
            "Expected a " + ConstantKey.class + " for value, but got: " + valueIK + ".");
    return ret;
  }
}
