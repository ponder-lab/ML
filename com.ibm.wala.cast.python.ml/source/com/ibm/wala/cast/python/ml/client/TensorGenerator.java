package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT_OP_CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONVERT_TO_TENSOR_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FIELD_REFERENCE_TO_DTYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.MODEL;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NDARRAY_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSORFLOW;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSOR_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.VARIABLES_VARIABLE;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.getFunction;
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
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
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
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashMapFactory;
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

  protected static final int UNDEFINED_PARAMETER_POSITION = -1;

  private static final Logger LOGGER = Logger.getLogger(TensorGenerator.class.getName());

  protected PointsToSetVariable source;

  protected final java.util.Map<InstanceKey, Set<List<Dimension<?>>>> shapeCache;

  protected final java.util.Map<InstanceKey, Set<DType>> dtypeCache;

  public TensorGenerator(PointsToSetVariable source) {
    this.source = source;
    this.shapeCache = HashMapFactory.make();
    this.dtypeCache = HashMapFactory.make();
  }

  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    try {
      Set<List<Dimension<?>>> shapes = this.getShapes(builder);
      Set<DType> dTypes = this.getDTypes(builder);

      Set<TensorType> ret = HashSetFactory.make();

      // Create a tensor type for each possible shape and dtype combination.
      for (List<Dimension<?>> dimensionList : shapes)
        for (DType dtype : dTypes)
          ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));

      LOGGER.fine(() -> "Generator " + this.getClass().getSimpleName() + " produced types: " + ret);

      return ret;
    } finally {
      this.clearCaches();
    }
  }

  private void clearCaches() {
    this.shapeCache.clear();
    this.dtypeCache.clear();
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
      if (this.shapeCache.containsKey(instanceKey)) {
        ret.addAll(this.shapeCache.get(instanceKey));
        continue;
      }

      Set<List<Dimension<?>>> ikShapes = HashSetFactory.make();
      AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
      if (asin == null) continue;
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
          Integer fieldIndex = TensorGenerator.getFieldIndex(constantKey);

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
              Number shapeValue = (Number) instanceFieldValue;
              LOGGER.fine(
                  "Found shape value: "
                      + shapeValue
                      + " for "
                      + this.getSource().getPointerKey()
                      + ".");

              Dimension<Integer> dimension =
                  (shapeValue != null) ? new NumericDim(shapeValue.intValue()) : null;

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

            ikShapes.add(asList(dimensions));
          }
      } else if (reference.equals(CONSTANT_OP_CONSTANT)) {
        FieldReference valueField =
            FieldReference.findOrCreate(CONSTANT_OP_CONSTANT, findOrCreateAsciiAtom("value"), Root);
        IField f = builder.getClassHierarchy().resolveField(valueField);
        PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
        OrdinalSet<InstanceKey> valuePts = pointerAnalysis.getPointsToSet(pk);
        ikShapes.addAll(this.getShapesFromShapeArgument(builder, valuePts));
      } else
        throw new IllegalStateException(
            "Expected a " + list + " or " + tuple + " for the shape, but got: " + reference + ".");

      this.shapeCache.put(instanceKey, ikShapes);
      ret.addAll(ikShapes);
    }

    return ret;
  }

  private static Integer getFieldIndex(ConstantKey<?> constantKey) {
    Object constantKeyValue = constantKey.getValue();

    if (constantKeyValue instanceof Integer) return (Integer) constantKeyValue;
    else if (constantKeyValue instanceof String) return Integer.parseInt((String) constantKeyValue);

    return null;
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

  /**
   * Returns the position of the shape parameter in the TensorFlow function.
   *
   * @return The position of the shape parameter in the TensorFlow function or a number less than 0
   *     if there is no shape parameter.
   */
  protected abstract int getShapeParameterPosition();

  /**
   * Returns the name of the shape parameter in the TensorFlow function.
   *
   * @return The name of the shape parameter in the TensorFlow function or <code>null</code> if
   *     there is no shape parameter.
   */
  protected abstract String getShapeParameterName();

  /**
   * Returns the possible shapes of the tensor returned by this generator.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return a set of shapes, where each shape is represented as a list of dimensions
   */
  protected Set<List<Dimension<?>>> getShapes(PropagationCallGraphBuilder builder) {
    int valNum =
        this.getArgumentValueNumber(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName(), true);
    if (valNum <= 0) return this.getDefaultShapes(builder);

    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getShapeParameterPosition(), this.getShapeParameterName());

    // If the argument shape is not specified.
    if (pointsToSet == null || pointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for shape argument in source: " + this.getSource() + ".");
    else
      // The shape points-to set is non-empty, meaning that the shape was explicitly set.
      return this.getShapesFromShapeArgument(builder, pointsToSet);
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
    return this.getShapesOfValue(builder, valuePointsToSet);
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

    for (InstanceKey valueIK : valuePointsToSet) {
      if (this.shapeCache.containsKey(valueIK)) {
        ret.addAll(this.shapeCache.get(valueIK));
        continue;
      }

      Set<List<Dimension<?>>> ikShapes = HashSetFactory.make();
      if (valueIK instanceof ConstantKey) ikShapes.add(emptyList()); // Scalar value.
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
            Integer fieldIndex = getFieldIndex(constantKey);

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
                this.getShapesOfValue(builder, instanceFieldPointsToSet);

            for (List<Dimension<?>> shapeList : shapesOfField) {
              List<Dimension<?>> shape = new ArrayList<>();

              shape.add(new NumericDim(objectCatalogPointsToSet.size()));
              shape.addAll(shapeList);

              ikShapes.add(shape);
            }
          }
        } else if (reference.equals(CONSTANT_OP_CONSTANT)) {
          FieldReference valueField =
              FieldReference.findOrCreate(
                  CONSTANT_OP_CONSTANT, findOrCreateAsciiAtom("value"), Root);
          IField f = builder.getClassHierarchy().resolveField(valueField);
          PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
          OrdinalSet<InstanceKey> valuePts = pointerAnalysis.getPointsToSet(pk);
          ikShapes.addAll(this.getShapesOfValue(builder, valuePts));
        } else if (reference.equals(VARIABLES_VARIABLE)) {
          FieldReference shapeField =
              FieldReference.findOrCreate(VARIABLES_VARIABLE, findOrCreateAsciiAtom("shape"), Root);
          IField f = builder.getClassHierarchy().resolveField(shapeField);
          PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
          OrdinalSet<InstanceKey> shapePts = pointerAnalysis.getPointsToSet(pk);

          if (shapePts != null && !shapePts.isEmpty()) {
            ikShapes.addAll(this.getShapesFromShapeArgument(builder, shapePts));
          } else {
            // Fallback to initial_value.
            FieldReference valField =
                FieldReference.findOrCreate(
                    VARIABLES_VARIABLE, findOrCreateAsciiAtom("initial_value"), Root);
            f = builder.getClassHierarchy().resolveField(valField);
            pk = builder.getPointerKeyForInstanceField(asin, f);
            OrdinalSet<InstanceKey> valPts = pointerAnalysis.getPointsToSet(pk);
            if (valPts != null && !valPts.isEmpty()) {
              ikShapes.addAll(this.getShapesOfValue(builder, valPts));
            }
          }
        } else if (reference.equals(TENSOR_TYPE)
            || reference.equals(CONVERT_TO_TENSOR_TYPE)
            || reference.equals(NDARRAY_TYPE)
            || reference.equals(MODEL.getDeclaringClass())) {
          // Already a tensor or a model, do nothing. Shapes will flow via the dataflow graph.
          LOGGER.fine(
              "Encountered " + reference.getName() + ". Shape will flow via dataflow graph.");
        } else throw new IllegalStateException("Unknown type reference: " + reference + ".");
      } else throw new IllegalStateException("Unknown value type: " + valueIK.getClass() + ".");

      this.shapeCache.put(valueIK, ikShapes);
      ret.addAll(ikShapes);
    }

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
  protected Set<DType> getDTypesFromDTypeArgument(
      PropagationCallGraphBuilder builder, Iterable<InstanceKey> pointsToSet) {
    if (pointsToSet == null || !pointsToSet.iterator().hasNext())
      throw new IllegalArgumentException(
          "Empty points-to set for dtype argument in source: " + this.getSource() + ".");

    Set<DType> ret = EnumSet.noneOf(DType.class);
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
        boolean found = false;
        CGNode importNode = null;

        if (instanceKey instanceof AllocationSiteInNode)
          importNode = ((AllocationSiteInNode) instanceKey).getNode();

        if (importNode == null)
          throw new IllegalStateException("Cannot find import node for DType: " + instanceKey);

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

        if (!found) throw new IllegalStateException("Unknown dtype: " + instanceKey + ".");
      } else if (instanceKey instanceof ConstantKey
          && ((ConstantKey<?>) instanceKey).getValue() instanceof String) {
        String value = (String) ((ConstantKey<?>) instanceKey).getValue();
        DType dtype = null;

        try {
          dtype = DType.valueOf(value.toUpperCase()); // Validate the dtype string.
        } catch (IllegalArgumentException | NullPointerException e) {
          throw new IllegalStateException("Unknown dtype string: " + value + ".", e);
        }

        ret.add(dtype);
        LOGGER.info(
            "Found dtype: "
                + dtype
                + " for source: "
                + this.getSource()
                + " from string: "
                + value
                + ".");
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
  protected abstract Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder);

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

  protected abstract String getDTypeParameterName();

  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder) {
    int valNum =
        this.getArgumentValueNumber(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName(), true);
    if (valNum <= 0) return this.getDefaultDTypes(builder);

    OrdinalSet<InstanceKey> pointsToSet =
        this.getArgumentPointsToSet(
            builder, this.getDTypeParameterPosition(), this.getDTypeParameterName());

    // If the argument dtype is not specified.
    if (pointsToSet == null || pointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for dtype argument in source: " + this.getSource() + ".");
    else
      // The dtype points-to set is non-empty, meaning that the dtype was explicitly set.
      return this.getDTypesFromDTypeArgument(builder, pointsToSet);
  }

  /**
   * Returns the possible dtypes of the tensor returned by this generator. The dtype is inferred
   * from the argument represented by the given value number.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param valueNumber The value number of the argument from which to infer the dtype.
   * @return A set of possible dtypes of the tensor returned by this generator.
   */
  protected Set<DType> getDTypes(PropagationCallGraphBuilder builder, int valueNumber) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    PointerKey valuePK =
        pointerAnalysis.getHeapModel().getPointerKeyForLocal(this.getNode(), valueNumber);
    OrdinalSet<InstanceKey> valuePointsToSet = pointerAnalysis.getPointsToSet(valuePK);

    if (valuePointsToSet == null || valuePointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for value number: " + valueNumber + " in: " + this.getNode() + ".");

    return this.getDTypesOfValue(builder, valuePointsToSet);
  }

  /**
   * Returns the possible dtypes of the tensor returned by this generator. The dtype is inferred
   * from the given points-to set.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param pointsToSet The points-to set of the value from which the dtype will be derived.
   * @return A set of possible dtypes of the tensor returned by this generator.
   */
  protected Set<DType> getDTypesOfValue(
      PropagationCallGraphBuilder builder, OrdinalSet<InstanceKey> valuePointsToSet) {
    if (valuePointsToSet == null || valuePointsToSet.isEmpty())
      throw new IllegalArgumentException(
          "Empty points-to set for value in source: " + this.getSource() + ".");

    Set<DType> ret = EnumSet.noneOf(DType.class);
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();

    for (InstanceKey valueIK : valuePointsToSet) {
      if (this.dtypeCache.containsKey(valueIK)) {
        ret.addAll(this.dtypeCache.get(valueIK));
        continue;
      }

      Set<DType> ikDTypes = EnumSet.noneOf(DType.class);
      if (valueIK instanceof ConstantKey) { // It's a scalar value.
        ConstantKey<?> constantKey = (ConstantKey<?>) valueIK;
        Object value = constantKey.getValue();
        if (value instanceof Float || value instanceof Double) {
          ikDTypes.add(FLOAT32);
          LOGGER.info(
              "Inferred dtype: "
                  + FLOAT32
                  + " for source: "
                  + this.getSource()
                  + " from value: "
                  + value
                  + ".");
        } else if (value instanceof Integer || value instanceof Long) {
          ikDTypes.add(INT32);
          LOGGER.info(
              "Inferred dtype: "
                  + INT32
                  + " for source: "
                  + this.getSource()
                  + " from value: "
                  + value
                  + ".");
        } else if (value instanceof String) {
          ikDTypes.add(STRING);
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
            Integer fieldIndex = getFieldIndex(constantKey);

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

            ikDTypes.addAll(this.getDTypesOfValue(builder, instanceFieldPointsToSet));
          }
        } else if (reference.equals(CONSTANT_OP_CONSTANT)) {
          FieldReference valueField =
              FieldReference.findOrCreate(
                  CONSTANT_OP_CONSTANT, findOrCreateAsciiAtom("value"), Root);
          IField f = builder.getClassHierarchy().resolveField(valueField);
          PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
          OrdinalSet<InstanceKey> valuePts = pointerAnalysis.getPointsToSet(pk);
          ikDTypes.addAll(this.getDTypesOfValue(builder, valuePts));
        } else if (reference.equals(VARIABLES_VARIABLE)) {
          FieldReference dtypeField =
              FieldReference.findOrCreate(VARIABLES_VARIABLE, findOrCreateAsciiAtom("dtype"), Root);
          IField f = builder.getClassHierarchy().resolveField(dtypeField);
          PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
          OrdinalSet<InstanceKey> dtypePts = pointerAnalysis.getPointsToSet(pk);

          if (dtypePts != null && !dtypePts.isEmpty()) {
            ikDTypes.addAll(this.getDTypesFromDTypeArgument(builder, dtypePts));
          } else {
            // Fallback to initial_value.
            FieldReference valField =
                FieldReference.findOrCreate(
                    VARIABLES_VARIABLE, findOrCreateAsciiAtom("initial_value"), Root);
            f = builder.getClassHierarchy().resolveField(valField);
            pk = builder.getPointerKeyForInstanceField(asin, f);
            OrdinalSet<InstanceKey> valPts = pointerAnalysis.getPointsToSet(pk);
            if (valPts != null && !valPts.isEmpty()) {
              ikDTypes.addAll(this.getDTypesOfValue(builder, valPts));
            }
          }
        } else if (reference.equals(TENSOR_TYPE)
            || reference.equals(CONVERT_TO_TENSOR_TYPE)
            || reference.equals(NDARRAY_TYPE)
            || reference.equals(MODEL.getDeclaringClass())) {
          // Already a tensor or a model, do nothing. DTypes will flow via the dataflow graph.
          LOGGER.fine(
              "Encountered " + reference.getName() + ". DType will flow via dataflow graph.");
        } else throw new IllegalStateException("Unknown type reference: " + reference + ".");
      } else throw new IllegalStateException("Unknown value type: " + valueIK.getClass() + ".");

      this.dtypeCache.put(valueIK, ikDTypes);
      ret.addAll(ikDTypes);
    }

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
    if (parameterPosition < 0) return UNDEFINED_PARAMETER_POSITION; // No such argument.

    int index = this.getNode().getMethod().isStatic() ? parameterPosition : parameterPosition + 1;

    if (index >= this.getNode().getIR().getNumberOfParameters())
      return UNDEFINED_PARAMETER_POSITION;

    return this.getNode().getIR().getParameter(index);
  }

  /**
   * Returns the points-to set of the argument at the specified position or with the specified name.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The position of the argument in the function call.
   * @param paramName The name of the argument in the function call.
   * @return The points-to set of the argument at the specified position or with the specified name.
   */
  protected OrdinalSet<InstanceKey> getArgumentPointsToSet(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    // 1. Try argument from callers (keyword or positional) - This is more precise for
    // context-sensitive nodes
    CallString cs = (CallString) this.getNode().getContext().get(CALL_STRING);
    if (cs != null) {
      OrdinalSet<InstanceKey> combinedPts = OrdinalSet.empty();
      boolean found = false;

      for (int i = 0; i < cs.getCallSiteRefs().length; i++) {
        CallSiteReference siteReference = cs.getCallSiteRefs()[i];
        IMethod callerMethod = cs.getMethods()[i];

        for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(this.getNode());
            it.hasNext(); ) {
          CGNode caller = it.next();

          // Only consider the caller that matches the context
          if (!caller.getMethod().equals(callerMethod)) {
            continue;
          }

          SSAAbstractInvokeInstruction[] calls = caller.getIR().getCalls(siteReference);
          for (SSAAbstractInvokeInstruction callInstr : calls) {
            // Verify this specific call instruction actually targets our node in this context
            boolean targetsThisNode = false;
            for (CGNode target : builder.getCallGraph().getPossibleTargets(caller, siteReference)) {
              if (target.equals(this.getNode())) {
                targetsThisNode = true;
                break;
              }
            }

            if (!targetsThisNode) {
              continue;
            }

            if (callInstr instanceof PythonInvokeInstruction) {
              PythonInvokeInstruction pyCallInstr = (PythonInvokeInstruction) callInstr;
              int argValNum = -1;

              if (paramName != null) {
                argValNum = pyCallInstr.getUse(paramName);
              }

              if (argValNum == -1 && paramPos >= 0) {
                int numPosParams =
                    pyCallInstr.getNumberOfPositionalParameters() - 1; // Exclude function.
                if (paramPos < numPosParams) {
                  argValNum =
                      pyCallInstr.getUse(paramPos + 1); // Positional arguments start at index 1.
                }
              }

              if (argValNum != -1) {
                PointerKey argPk =
                    builder
                        .getPointerAnalysis()
                        .getHeapModel()
                        .getPointerKeyForLocal(caller, argValNum);
                OrdinalSet<InstanceKey> argPts = builder.getPointerAnalysis().getPointsToSet(argPk);
                if (argPts != null && !argPts.isEmpty()) {
                  combinedPts = OrdinalSet.unify(combinedPts, argPts);
                  found = true;
                }
              }
            }
          }
        }
      }

      if (found) {
        return combinedPts;
      }
    }

    // 2. Fallback: Try positional parameter in callee
    int valNum = this.getArgumentValueNumber(builder, paramPos, paramName, true);
    if (valNum > 0) {
      PointerKey pk =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(getNode(), valNum);
      OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(pk);
      if (pts != null && !pts.isEmpty()) {
        return pts;
      }
    }

    return OrdinalSet.empty();
  }

  /**
   * Returns the value number for the argument at the specified position or with the specified name.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The position of the argument in the function call.
   * @param paramName The name of the argument in the function call.
   * @param optional Whether the argument is optional.
   * @return The value number for the argument at the specified position or with the specified name
   *     or -1 if the argument is optional and not present.
   * @throws IllegalStateException If the argument is mandatory and not present.
   */
  protected int getArgumentValueNumber(
      PropagationCallGraphBuilder builder, int paramPos, String paramName, boolean optional) {
    Set<Integer> numArgs = this.getNumberOfPossiblePositionalArguments(builder);

    boolean keywordPresent =
        (paramName != null && this.isKeywordArgumentPresent(builder, paramName));
    boolean positionalPresent = numArgs.stream().anyMatch(n -> n > paramPos);

    if (!positionalPresent && !keywordPresent)
      if (optional) return -1;
      else
        throw new IllegalStateException(
            "Cannot determine value number for parameter at position "
                + paramPos
                + (paramName == null ? "" : " or name " + paramName)
                + " of "
                + this.getSignature());

    return this.getArgumentValueNumber(paramPos);
  }

  /**
   * Returns whether the keyword argument with the specified name is present in the function call.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramName The name of the keyword argument.
   * @return {@code true} if the keyword argument is present, {@code false} otherwise.
   */
  protected boolean isKeywordArgumentPresent(
      PropagationCallGraphBuilder builder, String paramName) {
    CallString cs = (CallString) this.getNode().getContext().get(CALL_STRING);
    if (cs == null || cs.getCallSiteRefs().length == 0) return false;

    CallSiteReference siteReference = cs.getCallSiteRefs()[0];

    for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(this.getNode());
        it.hasNext(); ) {
      CGNode caller = it.next();
      SSAAbstractInvokeInstruction[] calls = caller.getIR().getCalls(siteReference);

      for (SSAAbstractInvokeInstruction callInstr : calls) {
        if (callInstr instanceof PythonInvokeInstruction) {
          PythonInvokeInstruction pyCallInstr = (PythonInvokeInstruction) callInstr;
          if (pyCallInstr.getKeywords().contains(paramName)) {
            return true;
          }
        }
      }
    }
    return false;
  }

  /**
   * Returns the value number for the argument at the specified position.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The position of the argument in the function call.
   * @param optional Whether the argument is optional.
   * @return The value number for the argument at the specified position or -1 if the argument is
   *     optional and not present.
   * @throws IllegalStateException If the argument is mandatory and not present.
   */
  protected int getArgumentValueNumber(
      PropagationCallGraphBuilder builder, int paramPos, boolean optional) {
    return this.getArgumentValueNumber(builder, paramPos, null, optional);
  }

  /**
   * Returns the value number for the argument at the specified position.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The position of the argument in the function call.
   * @return The value number for the argument at the specified position.
   * @throws IllegalStateException If the argument is mandatory and not present.
   */
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
    if (cs == null || cs.getCallSiteRefs().length == 0) return ret;

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

        if (callInstr instanceof PythonInvokeInstruction) {
          PythonInvokeInstruction pyCallInstr = (PythonInvokeInstruction) callInstr;
          int numberOfPositionalParameters =
              pyCallInstr.getNumberOfPositionalParameters() - 1; // Exclude the function name.
          LOGGER.finer(
              () -> "Number of positional parameters: " + numberOfPositionalParameters + ".");

          ret.add(numberOfPositionalParameters);
        }
      }
    }

    return ret;
  }

  /**
   * Returns the possible double values for the given value number.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param caller The {@link CGNode} calling the function.
   * @param vn The value number of the argument.
   * @return A set of possible double values.
   */
  protected static Set<Double> getPossibleDoubleValues(
      PropagationCallGraphBuilder builder, CGNode caller, int vn) {
    Set<Double> vals = HashSetFactory.make();
    if (vn == -1) return vals;

    // 1. Try symbol table (for literal constants)
    if (caller.getIR().getSymbolTable().isConstant(vn)) {
      Object val = caller.getIR().getSymbolTable().getConstantValue(vn);
      if (val instanceof Number) {
        vals.add(((Number) val).doubleValue());
      }
    }

    // 2. Try points-to analysis
    PointerKey pk = builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(caller, vn);
    vals.addAll(getPossibleDoubleValues(builder.getPointerAnalysis().getPointsToSet(pk)));

    return vals;
  }

  /**
   * Returns the possible double values for the given points-to set.
   *
   * @param pts The points-to set of the argument.
   * @return A set of possible double values.
   */
  protected static Set<Double> getPossibleDoubleValues(OrdinalSet<InstanceKey> pts) {
    Set<Double> ret = HashSetFactory.make();

    for (Object val : getConstantValues(pts, true)) {
      if (val instanceof Number) {
        ret.add(((Number) val).doubleValue());
      } else if (val == null) {
        ret.add(null);
      } else {
        throw new IllegalStateException("Expected a number but found: " + val.getClass() + ".");
      }
    }

    return ret;
  }

  /**
   * Returns the possible long values for the given points-to set. If the value is `None`, then a
   * null value will be contained within the returned set.
   *
   * @param pointsToSet The points-to set of the value.
   * @return A set of possible long values. If the value is `None`, then a null value will be
   *     contained within the returned set.
   */
  protected static Set<Long> getPossibleLongValues(OrdinalSet<InstanceKey> pointsToSet) {
    Set<Long> ret = HashSetFactory.make();

    for (Object val : getConstantValues(pointsToSet, true)) {
      if (val instanceof Number) {
        ret.add(((Number) val).longValue());
      } else if (val == null) {
        ret.add(null);
      } else {
        throw new IllegalStateException("Expected a number but found: " + val.getClass() + ".");
      }
    }

    return ret;
  }

  /**
   * Returns a set of constant values derived from the given points-to set.
   *
   * @param pts The points-to set to analyze.
   * @param requireConstants If true, throws an exception if a non-constant key is encountered.
   * @return A set of constant values (which may contain nulls).
   * @throws IllegalStateException If {@code requireConstants} is true and a non-constant key is
   *     found.
   */
  protected static Set<Object> getConstantValues(
      OrdinalSet<InstanceKey> pts, boolean requireConstants) {
    Set<Object> ret = HashSetFactory.make();

    if (pts != null) {
      for (InstanceKey ik : pts) {
        if (ik instanceof ConstantKey) {
          ret.add(((ConstantKey<?>) ik).getValue());
        } else if (requireConstants) {
          throw new IllegalStateException(
              "Expected a constant key but found: " + ik.getClass() + ".");
        }
      }
    }

    return ret;
  }
}
