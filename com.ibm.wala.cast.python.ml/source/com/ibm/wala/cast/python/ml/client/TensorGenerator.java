package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FIELD_REFERENCE_TO_DTYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSORFLOW;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.getFunction;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;
import static java.util.stream.Collectors.toSet;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IField;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.ConstantKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerAnalysis;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.types.Descriptor;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Logger;

public abstract class TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(TensorGenerator.class.getName());

  private static final MethodReference IMPORT =
      MethodReference.findOrCreate(
          TENSORFLOW,
          findOrCreateAsciiAtom("import"),
          Descriptor.findOrCreate(null, TENSORFLOW.getName()));

  protected PointsToSetVariable source;

  public TensorGenerator(PointsToSetVariable source) {
    this.source = source;
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
   * @param pointsToSet The points-to set of the shape argument. FIXME: Why not take a value number?
   * @return A set of possible shapes of the tensor returned by this generator.
   */
  protected Set<List<Dimension<?>>> getShapesFromShapeArgument(
      PropagationCallGraphBuilder builder, Iterable<InstanceKey> pointsToSet) {
    if (pointsToSet == null || !pointsToSet.iterator().hasNext())
      // TODO: The shape argument could be a tensor, in which case the points-to set would be empty.
      throw new IllegalArgumentException(
          "Empty points-to set for shape argument in source: " + this.getSource() + ".");

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey instanceKey : pointsToSet) {
      AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
      TypeReference reference = asin.getConcreteType().getReference();

      if (reference.equals(list) || reference.equals(tuple)) {
        // We have a list of integers that represent the shape.
        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        // We expect the object catalog to contain a list of integers. Each element in the array
        // corresponds to the set of possible dimensions for that index.
        @SuppressWarnings({"unchecked", "rawtypes"})
        Set<Dimension<Integer>>[] possibleDimensions = new Set[objectCatalogPointsToSet.size()];

        for (InstanceKey catalogIK : objectCatalogPointsToSet) {
          ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
          Object constantKeyValue = constantKey.getValue();

          Integer fieldIndex = (Integer) constantKeyValue;

          FieldReference subscript =
              FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

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
                  "Found shape value: "
                      + shapeValue
                      + " for "
                      + this.getSource().getPointerKey()
                      + ".");

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
                  + this.getSource()
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
            @SuppressWarnings({"unchecked", "rawtypes"})
            Dimension<Integer>[] dimensions = new Dimension[possibleDimensions.length];

            dimensions[i] = iDim;

            for (int j = 0; j < possibleDimensions.length; j++)
              if (i != j)
                for (Dimension<Integer> jDim : possibleDimensions[j]) dimensions[j] = jDim;

            ret.add(asList(dimensions));
          }
      } else
        throw new IllegalStateException(
            "Expected a "
                + PythonTypes.list
                + " or "
                + PythonTypes.tuple
                + " for the shape, but got: "
                + reference
                + ".");
    }

    return ret;
  }

  /**
   * Returns the default shapes if no shape argument is provided.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The default shapes if no shape argument is provided.
   */
  protected abstract Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder);

  /**
   * Returns the value number for the shape argument in the function call. A return value of a
   * number less than or equal to zero signifies that there is no shape parameter.
   *
   * @return The value number for the shape argument in the function call. May return a number less
   *     than or equal to 0 if there is no shape parameter.
   */
  protected int getShapeArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getShapeParameterPosition());
  }

  protected abstract int getShapeParameterPosition();

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
    int shapeArgValueNum = this.getShapeArgumentValueNumber();
    OrdinalSet<InstanceKey> pointsToSet = null;

    if (shapeArgValueNum > 0) {
      PointerKey pointerKey =
          pointerAnalysis.getHeapModel().getPointerKeyForLocal(getNode(), shapeArgValueNum);
      pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);
    }

    // If the argument shape is not specified.
    if (pointsToSet == null || pointsToSet.isEmpty()) return getDefaultShapes(builder);
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
    PointerKey valuePK =
        pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), valueNumber);
    OrdinalSet<InstanceKey> valuePointsToSet = pointerAnalysis.getPointsToSet(valuePK);

    if (valuePointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for value number: " + valueNumber + " in: " + this.getNode() + ".");

    // FIXME: Just use the value number directly?
    return getShapesOfValue(builder, valuePointsToSet);
  }

  /**
   * Returns the possible shapes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param pointsToSet The points-to set of the value from which the shape will be derived.
   * @return A set of possible shapes of the tensor returned by this generator.
   */
  protected Set<List<Dimension<?>>> getShapesOfValue(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    if (valuePointsToSet == null || valuePointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for value in source: " + this.getSource() + ".");

    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey valueIK : valuePointsToSet)
      if (valueIK instanceof ConstantKey) ret.add(emptyList()); // Scalar value.
      else if (valueIK instanceof AllocationSiteInNode) {
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
        TypeReference reference = asin.getConcreteType().getReference();

        if (reference.equals(list) || reference.equals(tuple)) {
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
                    Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

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
      } else throw new IllegalStateException("Unknown value type: " + valueIK.getClass() + ".");

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
  protected EnumSet<DType> getDTypesFromDTypeArgument(
      PropagationCallGraphBuilder builder, Iterable<InstanceKey> pointsToSet) {
    if (pointsToSet == null || !pointsToSet.iterator().hasNext())
      throw new IllegalArgumentException(
          "Empty points-to set for dtype argument in source: " + this.getSource() + ".");

    EnumSet<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey instanceKey : pointsToSet) {
      // First, check for `None`.
      if (instanceKey instanceof ConstantKey) {
        ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
        Object value = constantKey.getValue();

        if (value == null) {
          LOGGER.info(
              "DType argument is None for source: "
                  + this.getSource()
                  + "; using default dtypes."
                  + ".");
          return getDefaultDTypes(builder);
        }
      }

      IClass concreteType = instanceKey.getConcreteType();
      TypeReference typeReference = concreteType.getReference();

      if (typeReference.equals(TensorFlowTypes.D_TYPE)) {
        // we have a dtype.
        // let's see if it's a dtype.
        Set<CGNode> importNodes = builder.getCallGraph().getNodes(IMPORT);

        // find the import nodes from this file.
        Set<CGNode> importNodesOfInterest =
            importNodes.stream()
                .filter(
                    in -> {
                      CallString cs = (CallString) in.getContext().get(CALL_STRING);

                      // We expect the first method in the call string to be the import.
                      assert cs.getMethods().length == 1
                          : "Expected a single method in the call string, but got: "
                              + cs.getMethods().length
                              + " for node: "
                              + in;

                      IMethod method = cs.getMethods()[0];

                      CallString nodeCS = (CallString) this.getNode().getContext().get(CALL_STRING);

                      // We expect the first method in the call string to be the import.
                      assert nodeCS.getMethods().length == 1
                          : "Expected a single method in the call string, but got: "
                              + nodeCS.getMethods().length
                              + " for node: "
                              + in;

                      return method.equals(nodeCS.getMethods()[0]);
                    })
                .collect(toSet());

        if (importNodesOfInterest.isEmpty())
          throw new IllegalStateException(
              "No import nodes found for source: " + this.getSource() + ".");

        boolean found = false;

        for (CGNode importNode : importNodesOfInterest) {
          LOGGER.fine("Found import node of interest: " + importNode + ".");

          InstanceKey tensorFlowIK =
              pointerAnalysis
                  .getHeapModel()
                  .getInstanceKeyForAllocation(importNode, NewSiteReference.make(0, TENSORFLOW));

          // Check dtype literals.
          for (Entry<FieldReference, DType> entry : FIELD_REFERENCE_TO_DTYPE.entrySet()) {
            FieldReference fieldRef = entry.getKey();
            DType dtype = entry.getValue();
            IField field = builder.getClassHierarchy().resolveField(fieldRef);

            PointerKey pk =
                pointerAnalysis.getHeapModel().getPointerKeyForInstanceField(tensorFlowIK, field);

            for (InstanceKey ik : pointerAnalysis.getPointsToSet(pk))
              if (ik.equals(instanceKey)) {
                ret.add(dtype);
                LOGGER.info(
                    "Found dtype: "
                        + dtype
                        + " for source: "
                        + this.getSource()
                        + " from dType: "
                        + instanceKey
                        + ".");
                found = true;
              }
          }
        }

        if (!found) throw new IllegalStateException("Unknown dtype: " + instanceKey + ".");
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
  protected int getDTypeArgumentValueNumber() {
    return this.getArgumentValueNumber(this.getDTypeParameterPosition());
  }

  protected abstract int getDTypeParameterPosition();

  protected EnumSet<DType> getDTypes(PropagationCallGraphBuilder builder) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    int valNum = this.getDTypeArgumentValueNumber();
    OrdinalSet<InstanceKey> pointsToSet = null;

    if (valNum > 0) {
      // The dtype is in an explicit argument.
      // FIXME: Handle keyword arguments.
      PointerKey pointerKey =
          pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), valNum);
      pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);
    }

    // If the argument dtype is not specified.
    if (pointsToSet == null || pointsToSet.isEmpty()) return getDefaultDTypes(builder);
    else
      // The dtype points-to set is non-empty, meaning that the dtype was explicitly set.
      return getDTypesFromDTypeArgument(builder, pointsToSet);
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
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    PointerKey valuePK =
        pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), valueNumber);
    OrdinalSet<InstanceKey> valuePointsToSet = pointerAnalysis.getPointsToSet(valuePK);

    if (valuePointsToSet == null || valuePointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for value number: " + valueNumber + " in: " + this.getNode() + ".");

    return getDTypesOfValue(builder, valuePointsToSet);
  }

  /**
   * Returns the possible dtypes of the tensor returned by this generator. The dtype is inferred
   * from the given points-to set.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param pointsToSet The points-to set of the value from which the dtype will be derived.
   * @return A set of possible dtypes of the tensor returned by this generator.
   */
  private EnumSet<DType> getDTypesOfValue(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    if (valuePointsToSet == null || valuePointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for value in source: " + this.getSource() + ".");

    EnumSet<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey valueIK : valuePointsToSet)
      if (valueIK instanceof ConstantKey) { // It's a scalar value.
        ConstantKey<?> constantKey = (ConstantKey<?>) valueIK;
        Object value = constantKey.getValue();
        if (value instanceof Float || value instanceof Double) {
          ret.add(FLOAT32);
          LOGGER.info(
              "Inferred dtype: "
                  + FLOAT32
                  + " for source: "
                  + this.getSource()
                  + " from value: "
                  + value
                  + ".");
        } else if (value instanceof Integer || value instanceof Long) {
          ret.add(INT32);
          LOGGER.info(
              "Inferred dtype: "
                  + INT32
                  + " for source: "
                  + this.getSource()
                  + " from value: "
                  + value
                  + ".");
        } else if (value instanceof String) {
          ret.add(STRING);
          LOGGER.info(
              "Inferred dtype: "
                  + STRING
                  + " for source: "
                  + this.getSource()
                  + " from value: "
                  + value
                  + ".");
        } else if (value != null)
          throw new IllegalStateException("Unknown constant type: " + value.getClass() + ".");
      } else if (valueIK instanceof AllocationSiteInNode) {
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);
        TypeReference reference = asin.getConcreteType().getReference();

        if (reference.equals(list) || reference.equals(tuple)) {
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
                    Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

            IField f = builder.getClassHierarchy().resolveField(subscript);
            LOGGER.fine("Found field: " + f);

            PointerKey pointerKeyForInstanceField = builder.getPointerKeyForInstanceField(asin, f);
            LOGGER.fine(
                "Found pointer key for instance field: " + pointerKeyForInstanceField + ".");

            OrdinalSet<InstanceKey> instanceFieldPointsToSet =
                pointerAnalysis.getPointsToSet(pointerKeyForInstanceField);
            LOGGER.fine("Points-to set for instance field: " + instanceFieldPointsToSet + ".");

            EnumSet<DType> dTypesOfField = getDTypesOfValue(builder, instanceFieldPointsToSet);
            ret.addAll(dTypesOfField);
          }
        } else throw new IllegalStateException("Unknown type reference: " + reference + ".");
      } else
        // TODO: More cases.
        throw new IllegalStateException(
            "Expected a " + ConstantKey.class + " for value, but got: " + valueIK + ".");
    return ret;
  }

  protected PointsToSetVariable getSource() {
    return this.source;
  }

  protected CGNode getNode() {
    return ((LocalPointerKey) this.getSource().getPointerKey()).getNode();
  }

  @Override
  public String toString() {
    return this.getSignature();
  }

  /**
   * Returns the TensorFlow function signature represented by this generator.
   *
   * @return The TensorFlow function signature represented by this generator.
   */
  protected String getSignature() {
    TypeReference function = getFunction(this.getSource());
    return TYPE_REFERENCE_TO_SIGNATURE.get(function);
  }

  protected int getArgumentValueNumber(int parameterPosition) {
    if (parameterPosition < 0) return -1; // No such argument.

    return this.getNode().getMethod().isStatic()
        ? this.getNode().getIR().getParameter(parameterPosition)
        : this.getNode().getIR().getParameter(parameterPosition + 1);
  }

  protected int getArgumentValueNumber(
      PropagationCallGraphBuilder builder, int paramPos, boolean optional) {
    Set<Integer> numArgs = this.getNumberOfPossiblePositionalArguments(builder);

    if (!numArgs.stream().anyMatch(n -> n >= paramPos + 1))
      if (optional) return -1;
      else
        throw new IllegalStateException(
            "Cannot determine value number for parameter at position "
                + paramPos
                + " of "
                + this.getSignature());

    return this.getArgumentValueNumber(paramPos);
  }

  protected int getArgumentValueNumber(PropagationCallGraphBuilder builder, int paramPos) {
    return this.getArgumentValueNumber(builder, paramPos, false);
  }

  /**
   * Returns the set of possible numbers of positional arguments passed to the range function at the
   * call.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used for the analysis.
   * @return A set of integers representing the possible number of positional arguments.
   */
  protected Set<Integer> getNumberOfPossiblePositionalArguments(
      PropagationCallGraphBuilder builder) {
    Set<Integer> ret = HashSetFactory.make();

    CallString cs = (CallString) this.getNode().getContext().get(CALL_STRING);
    CallSiteReference siteReference = cs.getCallSiteRefs()[0];
    LOGGER.fine(() -> "Analyzing call site: " + siteReference + ".");

    for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(this.getNode());
        it.hasNext(); ) {
      CGNode caller = it.next();
      LOGGER.fine(() -> "Analyzing caller node: " + caller.getMethod().getSignature() + ".");

      SSAAbstractInvokeInstruction[] calls = caller.getIR().getCalls(siteReference);
      LOGGER.finest(() -> "Number of calls at this site: " + calls.length + ".");

      for (SSAAbstractInvokeInstruction callInstr : calls) {
        LOGGER.finest(() -> "Call instruction: " + callInstr + ".");

        PythonInvokeInstruction pyCallInstr = (PythonInvokeInstruction) callInstr;
        int numberOfPositionalParameters =
            pyCallInstr.getNumberOfPositionalParameters() - 1; // Exclude the function name.
        LOGGER.finer(
            () -> "Number of positional parameters: " + numberOfPositionalParameters + ".");

        ret.add(numberOfPositionalParameters);
      }
    }

    return ret;
  }

  protected Set<Long> getPossibleLongArguments(
      PropagationCallGraphBuilder builder, int valueNumber) {
    Set<Long> ret = HashSetFactory.make();

    if (valueNumber >= 0) {
      PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
      PointerKey pointerKey =
          pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), valueNumber);
      OrdinalSet<InstanceKey> pointsToSet = pointerAnalysis.getPointsToSet(pointerKey);

      if (pointsToSet == null || pointsToSet.isEmpty())
        throw new IllegalArgumentException(
            "Empty points-to set in source: " + this.getSource() + ".");

      for (InstanceKey instanceKey : pointsToSet)
        if (instanceKey instanceof com.ibm.wala.ipa.callgraph.propagation.ConstantKey) {
          ConstantKey<?> constantKey = (ConstantKey<?>) instanceKey;
          Object constantKeyValue = constantKey.getValue();

          if (constantKeyValue instanceof Long) {
            Long value = (Long) constantKeyValue;
            ret.add(value);
          } else
            throw new IllegalStateException(
                "Expected a long, but found: " + constantKeyValue + ".");
        } else
          throw new IllegalStateException(
              "Expected a constant key, but found: " + instanceKey + ".");
    }

    return ret;
  }
}
