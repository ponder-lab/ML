package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ARRAY_OPS_RESHAPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.ARRAY_OPS_ZEROS;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT_OP_CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONVERT_TO_TENSOR_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FIELD_REFERENCE_TO_DTYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.LINALG_OPS_EYE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.NDARRAY_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.PLACEHOLDER;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_FACTORY_OPS_CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.RAGGED_MATH_OPS_RANGE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.SPARSE_TENSOR_TYPE;
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
import com.ibm.wala.ipa.callgraph.propagation.ReturnValueKey;
import com.ibm.wala.ipa.callgraph.propagation.cfa.CallString;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.debug.UnimplementedError;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * An abstract generator for {@link TensorType}s.
 *
 * <p>TODO: Revisit caching of shapes and dtypes.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public abstract class TensorGenerator {

  protected static final int UNDEFINED_PARAMETER_POSITION = -1;

  private static final Logger LOGGER = Logger.getLogger(TensorGenerator.class.getName());

  /** The source of the tensor, represented by a points-to set variable. */
  protected PointsToSetVariable source;

  /**
   * The call graph node representing the manual generator, used when standard points-to analysis
   * fails.
   */
  protected CGNode manualNode;

  public TensorGenerator(PointsToSetVariable source) {
    this.source = source;
  }

  public TensorGenerator(CGNode node) {
    this.manualNode = node;
  }

  /**
   * Returns a set of possible {@link TensorType}s that this generator can produce.
   *
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return A set of possible {@link TensorType}s.
   */
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> shapes = this.getShapes(builder);
    Set<DType> dTypes = this.getDTypes(builder);

    Set<TensorType> ret = HashSetFactory.make();

    // Create a tensor type for each possible shape and dtype combination.
    for (List<Dimension<?>> dimensionList : shapes)
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));

    LOGGER.fine(() -> "Generator " + this.getClass().getSimpleName() + " produced types: " + ret);

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
          Integer fieldIndex = getFieldIndex(constantKey);

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
                      + (this.getSource() != null ? this.getSource().getPointerKey() : "null")
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

            ret.add(asList(dimensions));
          }
      } else if (reference.equals(CONSTANT_OP_CONSTANT)) {
        // We have a constant tensor. We recurse into its value field.
        IField valueField =
            builder.getClassHierarchy().resolveField(TensorFlowTypes.CONSTANT_VALUE);
        PointerKey valuePK = builder.getPointerKeyForInstanceField(instanceKey, valueField);
        OrdinalSet<InstanceKey> valuePts = pointerAnalysis.getPointsToSet(valuePK);
        ret.addAll(this.getShapesFromShapeArgument(builder, valuePts));
      } else
        throw new IllegalStateException(
            "Expected a " + list + " or " + tuple + " for the shape, but got: " + reference + ".");
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
    if (pointsToSet == null || pointsToSet.isEmpty()) return getDefaultShapes(builder);
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

              ret.add(shape);
            }
          }
        } else if (reference.equals(TENSOR_TYPE)
            || reference.equals(CONVERT_TO_TENSOR_TYPE)
            || reference.equals(NDARRAY_TYPE)
            || reference.equals(TensorFlowTypes.OPERATION)
            || reference.equals(CONSTANT_OP_CONSTANT)
            || reference.equals(ARRAY_OPS_ZEROS)
            || reference.equals(ARRAY_OPS_RESHAPE)
            || reference.equals(VARIABLES_VARIABLE)
            || reference.equals(SPARSE_TENSOR_TYPE)
            || reference.equals(RAGGED_FACTORY_OPS_CONSTANT)
            || reference.equals(RAGGED_MATH_OPS_RANGE)
            || reference.equals(LINALG_OPS_EYE)
            || reference.equals(TensorFlowTypes.DATASET)
            || reference.equals(TensorFlowTypes.DATASET_SHUFFLE_TYPE)
            || reference.equals(TensorFlowTypes.DATASET_BATCH_TYPE)
            || reference.equals(TensorFlowTypes.DATASET_MAP_TYPE)
            || reference.equals(TensorFlowTypes.DATASET_RANGE_TYPE)
            || reference.equals(TensorFlowTypes.DATASET_FROM_TENSOR_SLICES_TYPE)
            || reference.equals(TensorFlowTypes.ADD.getDeclaringClass())
            || reference.equals(TensorFlowTypes.MULTIPLY.getDeclaringClass())
            || reference.equals(TensorFlowTypes.REDUCE_SUM.getDeclaringClass())
            || reference.equals(TensorFlowTypes.REDUCE_MEAN.getDeclaringClass())
            || reference.equals(TensorFlowTypes.ARGMAX.getDeclaringClass())
            || reference.equals(TensorFlowTypes.EQUAL.getDeclaringClass())) {
          // If the value is a tensor, we attempt to find the generator that created it and ask for
          // its shape.
          LOGGER.fine(
              "Encountered "
                  + reference.getName()
                  + ". Attempting to retrieve shape from producer.");
          ret.addAll(this.getShapesFromTensor(builder, asin));
        } else throw new IllegalStateException("Unknown type reference: " + reference + ".");
      } else throw new IllegalStateException("Unknown value type: " + valueIK.getClass() + ".");

    return ret;
  }

  private Set<List<Dimension<?>>> getShapesFromTensor(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    CGNode readDataNode = asin.getNode();

    // Support allocations directly in 'do' methods (preferred for 1-CFA context separation).
    if (readDataNode.getMethod().getName().toString().equals("do")) {
      int def = findDefinition(readDataNode, asin);
      if (def != -1) {
        PointerKey defKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(readDataNode, def);
        PointsToSetVariable defSource = null;
        try {
          defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
        } catch (UnimplementedError e) {
          // If the pointer key is implicit, we might fail to get the points-to set.
          LOGGER.log(Level.FINE, "Could not get points-to set for " + defKey, e);
          // Try to create a manual generator if possible.
        }

        TensorGenerator generator = null;
        TypeReference declaringClass = readDataNode.getMethod().getDeclaringClass().getReference();
        if (declaringClass.equals(PLACEHOLDER.getDeclaringClass())
            || declaringClass.equals(CONSTANT.getDeclaringClass())) {
          generator = createManualGenerator(readDataNode, builder);
        } else if (defSource != null) {
          // Avoid infinite recursion if the current generator is for the same source.
          if (this.getSource() != null && this.getSource().equals(defSource)) {
            return ret;
          }
          generator = TensorGeneratorFactory.getGenerator(defSource, builder);
        } else {
          generator = createManualGenerator(readDataNode, builder);
        }

        if (generator != null) {
          LOGGER.fine("Delegating shape inference to: " + generator);
          ret.addAll(generator.getShapes(builder));
        }
      }
      return ret;
    }

    // 1. read_data is called by the operation's 'do' method.
    Iterator<CGNode> doNodes = builder.getCallGraph().getPredNodes(readDataNode);
    while (doNodes.hasNext()) {
      CGNode doNode = doNodes.next();

      // 2. Find the instruction in 'do' that called 'read_data'.
      // We use the context of the callee (readDataNode) to identify the specific call site.
      CallString cs = (CallString) readDataNode.getContext().get(CALL_STRING);
      if (cs != null && cs.getCallSiteRefs().length > 0) {
        CallSiteReference readDataSite = cs.getCallSiteRefs()[0];
        IMethod callerMethod = cs.getMethods()[0];

        if (doNode.getMethod().equals(callerMethod)) {
          SSAAbstractInvokeInstruction[] calls = doNode.getIR().getCalls(readDataSite);
          for (SSAAbstractInvokeInstruction call : calls) {
            // Construct a source for the result of this call (which is the tensor object).
            if (call.getNumberOfDefs() > 0) {
              int def = call.getDef();
              PointerKey defKey =
                  builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(doNode, def);
              PointsToSetVariable defSource =
                  builder.getPropagationSystem().findOrCreatePointsToSet(defKey);

              // Instantiate the generator for this source.
              TensorGenerator generator = TensorGeneratorFactory.getGenerator(defSource, builder);
              LOGGER.fine("Delegating shape inference to: " + generator);
              ret.addAll(generator.getShapes(builder));
            }
          }
        }
      }
    }
    return ret;
  }

  private static int findDefinition(CGNode node, AllocationSiteInNode asin) {
    if (node.getIR() == null) return -1;
    for (SSAInstruction inst : node.getIR().getInstructions()) {
      if (inst != null && inst instanceof SSANewInstruction) {
        SSANewInstruction newInst = (SSANewInstruction) inst;
        if (newInst.getNewSite().equals(asin.getSite())) {
          return newInst.getDef();
        }
      }
    }
    return -1;
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

      // Check if it matches a known DType field (e.g., tf.float32).
      boolean found = false;
      if (instanceKey instanceof AllocationSiteInNode) {
        CGNode importNode = ((AllocationSiteInNode) instanceKey).getNode();

        if (importNode != null) {
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

            if (field != null) {
              PointerKey pk =
                  pointerAnalysis.getHeapModel().getPointerKeyForInstanceField(tensorFlowIK, field);

              OrdinalSet<InstanceKey> pts = pointerAnalysis.getPointsToSet(pk);
              if (pts != null) {
                for (InstanceKey ik : pts)
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
                    break;
                  }
              }
            }
            if (found) break;
          }
        }
      }

      if (found) continue;

      IClass concreteType = instanceKey.getConcreteType();
      TypeReference typeReference = concreteType.getReference();

      if (typeReference.equals(TensorFlowTypes.D_TYPE)) {
        throw new IllegalStateException("Unknown dtype: " + instanceKey + ".");
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
      } else {
        throw new IllegalStateException(
            "Expected a "
                + TensorFlowTypes.D_TYPE
                + " for the dtype, but got: "
                + typeReference
                + ".");
      }
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
    if (pointsToSet == null || pointsToSet.isEmpty()) return getDefaultDTypes(builder);
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

            ret.addAll(this.getDTypesOfValue(builder, instanceFieldPointsToSet));
          }
        } else if (reference.equals(TENSOR_TYPE)
            || reference.equals(CONVERT_TO_TENSOR_TYPE)
            || reference.equals(NDARRAY_TYPE)
            || reference.equals(TensorFlowTypes.OPERATION)
            || reference.equals(CONSTANT_OP_CONSTANT)
            || reference.equals(ARRAY_OPS_ZEROS)
            || reference.equals(ARRAY_OPS_RESHAPE)
            || reference.equals(VARIABLES_VARIABLE)
            || reference.equals(SPARSE_TENSOR_TYPE)
            || reference.equals(RAGGED_FACTORY_OPS_CONSTANT)
            || reference.equals(RAGGED_MATH_OPS_RANGE)
            || reference.equals(LINALG_OPS_EYE)
            || reference.equals(TensorFlowTypes.DATASET)
            || reference.equals(TensorFlowTypes.DATASET_SHUFFLE_TYPE)
            || reference.equals(TensorFlowTypes.DATASET_BATCH_TYPE)
            || reference.equals(TensorFlowTypes.DATASET_MAP_TYPE)
            || reference.equals(TensorFlowTypes.DATASET_RANGE_TYPE)
            || reference.equals(TensorFlowTypes.DATASET_FROM_TENSOR_SLICES_TYPE)
            || reference.equals(TensorFlowTypes.ADD.getDeclaringClass())
            || reference.equals(TensorFlowTypes.MULTIPLY.getDeclaringClass())
            || reference.equals(TensorFlowTypes.REDUCE_SUM.getDeclaringClass())
            || reference.equals(TensorFlowTypes.REDUCE_MEAN.getDeclaringClass())
            || reference.equals(TensorFlowTypes.ARGMAX.getDeclaringClass())
            || reference.equals(TensorFlowTypes.EQUAL.getDeclaringClass())) {
          // If the value is a tensor, we attempt to find the generator that created it and ask for
          // its dtype.
          LOGGER.fine(
              "Encountered "
                  + reference.getName()
                  + ". Attempting to retrieve dtype from producer.");
          ret.addAll(getDTypesFromTensor(builder, asin));
        } else throw new IllegalStateException("Unknown type reference: " + reference + ".");
      } else throw new IllegalStateException("Unknown value type: " + valueIK.getClass() + ".");
    }

    return ret;
  }

  private Set<DType> getDTypesFromTensor(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    CGNode readDataNode = asin.getNode();

    // Support allocations directly in 'do' methods (preferred for 1-CFA context separation).
    if (readDataNode.getMethod().getName().toString().equals("do")) {
      int def = findDefinition(readDataNode, asin);
      if (def != -1) {
        PointerKey defKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(readDataNode, def);
        PointsToSetVariable defSource = null;
        try {
          defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
        } catch (UnimplementedError e) {
          // If the pointer key is implicit, we might fail to get the points-to set.
          LOGGER.log(Level.FINE, "Could not get points-to set for " + defKey, e);
          // Try to create a manual generator if possible.
        }

        if (defSource != null) {
          TypeReference declaringClass =
              readDataNode.getMethod().getDeclaringClass().getReference();
          if (declaringClass.equals(PLACEHOLDER.getDeclaringClass())
              || declaringClass.equals(CONSTANT.getDeclaringClass())) {
            TensorGenerator generator = createManualGenerator(readDataNode, builder);
            if (generator != null) {
              LOGGER.fine("Delegating dtype inference to: " + generator);
              ret.addAll(generator.getDTypes(builder));
            }
          } else {
            if (this.getSource() != null && this.getSource().equals(defSource)) {
              return ret;
            }

            TensorGenerator generator = TensorGeneratorFactory.getGenerator(defSource, builder);
            LOGGER.fine("Delegating dtype inference to: " + generator);
            ret.addAll(generator.getDTypes(builder));
          }
        } else {
          TensorGenerator generator = createManualGenerator(readDataNode, builder);
          if (generator != null) {
            LOGGER.fine("Delegating dtype inference to: " + generator);
            ret.addAll(generator.getDTypes(builder));
          }
        }
      }
      return ret;
    }

    // Trace back to the user-level call that invoked this generator.
    // 1. read_data is called by the operation's 'do' method.
    Iterator<CGNode> doNodes = builder.getCallGraph().getPredNodes(readDataNode);
    while (doNodes.hasNext()) {
      CGNode doNode = doNodes.next();

      // 2. Find the instruction in 'do' that called 'read_data'.
      // We use the context of the callee (readDataNode) to identify the specific call site.
      CallString cs = (CallString) readDataNode.getContext().get(CALL_STRING);
      if (cs != null && cs.getCallSiteRefs().length > 0) {
        CallSiteReference readDataSite = cs.getCallSiteRefs()[0];
        IMethod callerMethod = cs.getMethods()[0];

        if (doNode.getMethod().equals(callerMethod)) {
          SSAAbstractInvokeInstruction[] calls = doNode.getIR().getCalls(readDataSite);
          for (SSAAbstractInvokeInstruction call : calls) {
            // Construct a source for the result of this call (which is the tensor object).
            if (call.getNumberOfDefs() > 0) {
              int def = call.getDef();
              PointerKey defKey =
                  builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(doNode, def);
              PointsToSetVariable defSource =
                  builder.getPropagationSystem().findOrCreatePointsToSet(defKey);

              // Instantiate the generator for this source.
              TensorGenerator generator = TensorGeneratorFactory.getGenerator(defSource, builder);
              LOGGER.fine("Delegating dtype inference to: " + generator);
              ret.addAll(generator.getDTypes(builder));
            }
          }
        }
      }
    }
    return ret;
  }

  protected PointsToSetVariable getSource() {
    return this.source;
  }

  protected CGNode getNode() {
    if (this.manualNode != null) {
      return this.manualNode;
    }
    PointerKey k = this.getSource().getPointerKey();
    if (k instanceof LocalPointerKey) {
      return ((LocalPointerKey) k).getNode();
    } else if (k instanceof ReturnValueKey) {
      return ((ReturnValueKey) k).getNode();
    }
    throw new IllegalArgumentException("Unsupported PointerKey type: " + k.getClass());
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
    TypeReference function;
    if (this.manualNode != null) {
      function = this.manualNode.getMethod().getDeclaringClass().getReference();
    } else {
      function = getFunction(this.getSource());
    }
    return TYPE_REFERENCE_TO_SIGNATURE.get(function);
  }

  protected static final int RECEIVER_PARAMETER_POSITION = -2;

  protected int getArgumentValueNumber(int parameterPosition) {
    if (parameterPosition == RECEIVER_PARAMETER_POSITION)
      return this.getNode().getIR().getParameter(0);
    if (parameterPosition < 0) return UNDEFINED_PARAMETER_POSITION; // No such argument.

    int index = this.getNode().getMethod().isStatic() ? parameterPosition : parameterPosition + 1;

    if (index >= this.getNode().getIR().getNumberOfParameters())
      return UNDEFINED_PARAMETER_POSITION;

    return this.getNode().getIR().getParameter(index);
  }

  protected PythonInvokeInstruction getInvokeInstruction() {
    if (this.source == null) {
      return null;
    }
    if (this.getSource().getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) this.getSource().getPointerKey();
      if (lpk.getNode().equals(this.getNode())) {
        int vn = lpk.getValueNumber();
        if (vn > 0) {
          SSAInstruction def = this.getNode().getDU().getDef(vn);
          if (def instanceof PythonInvokeInstruction) {
            return (PythonInvokeInstruction) def;
          }
        }
      }
    }
    return null;
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
    return getArgumentPointsToSet(builder, this.getNode(), paramPos, paramName);
  }

  protected OrdinalSet<InstanceKey> getArgumentPointsToSet(
      PropagationCallGraphBuilder builder, CGNode node, int paramPos, String paramName) {
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int argValNum = -1;

      if (paramName != null) {
        argValNum = call.getUse(paramName);
      }

      if (argValNum == -1 && paramPos >= 0) {

        int numKeywords = call.getKeywords() != null ? call.getKeywords().size() : 0;

        int numPosParams =
            call.getNumberOfUses() - 1 - numKeywords; // Exclude function and keywords.

        if (paramPos < numPosParams) {

          argValNum = call.getUse(paramPos + 1); // Positional arguments start at index 1.
        }
      }

      if (argValNum != -1) {
        PointerKey argPk =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, argValNum);
        OrdinalSet<InstanceKey> argPts = builder.getPointerAnalysis().getPointsToSet(argPk);
        if (argPts != null && !argPts.isEmpty()) {
          return argPts;
        }
      }
      return OrdinalSet.empty();
    }

    if (node.getMethod().getName().toString().equals("read_data")) {
      OrdinalSet<InstanceKey> ret = OrdinalSet.empty();
      Iterator<CGNode> preds = builder.getCallGraph().getPredNodes(node);
      while (preds.hasNext()) {
        CGNode pred = preds.next();
        if (pred.getMethod().getName().toString().equals("do")) {
          ret = OrdinalSet.unify(ret, getArgumentPointsToSet(builder, pred, paramPos, paramName));
        }
      }
      return ret;
    }

    // 1. Try argument from callers (keyword or positional) - This is more precise for
    // context-sensitive nodes
    CallString cs = (CallString) node.getContext().get(CALL_STRING);
    if (cs != null) {
      OrdinalSet<InstanceKey> combinedPts = OrdinalSet.empty();
      boolean found = false;
      boolean callAnalyzed = false;

      for (int i = 0; i < cs.getCallSiteRefs().length; i++) {
        CallSiteReference siteReference = cs.getCallSiteRefs()[i];
        IMethod callerMethod = cs.getMethods()[i];

        for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(node); it.hasNext(); ) {
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
              if (target.equals(node)) {
                targetsThisNode = true;
                break;
              }
            }

            if (!targetsThisNode) {
              continue;
            }
            callAnalyzed = true;

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
      if (callAnalyzed) {
        return OrdinalSet.empty();
      }
    }

    // 2. Fallback: Try positional parameter in callee
    int valNum = getArgumentValueNumber(node, paramPos);
    if (valNum > 0) {
      PointerKey pk =
          builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, valNum);
      OrdinalSet<InstanceKey> pts = builder.getPointerAnalysis().getPointsToSet(pk);
      if (pts != null && !pts.isEmpty()) {
        return pts;
      }
    }

    return OrdinalSet.empty();
  }

  private int getArgumentValueNumber(CGNode node, int parameterPosition) {
    if (parameterPosition == RECEIVER_PARAMETER_POSITION) return node.getIR().getParameter(0);
    if (parameterPosition < 0) return UNDEFINED_PARAMETER_POSITION; // No such argument.

    int index = node.getMethod().isStatic() ? parameterPosition : parameterPosition + 1;

    if (index >= node.getIR().getNumberOfParameters()) return UNDEFINED_PARAMETER_POSITION;

    return node.getIR().getParameter(index);
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
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int argValNum = -1;

      if (paramName != null) {
        argValNum = call.getUse(paramName);
      }

      if (argValNum == -1 && paramPos >= 0) {

        int numKeywords = call.getKeywords() != null ? call.getKeywords().size() : 0;

        int numPosParams =
            call.getNumberOfUses() - 1 - numKeywords; // Exclude function and keywords.

        if (paramPos < numPosParams) {

          argValNum = call.getUse(paramPos + 1); // Positional arguments start at index 1.
        }
      }

      if (argValNum != -1) return argValNum;

      if (optional) return -1;
      else
        throw new IllegalStateException(
            "Cannot determine value number for parameter at position "
                + paramPos
                + (paramName == null ? "" : " or name " + paramName)
                + " of "
                + this.getSignature());
    } else {
      // Fallback for manual nodes (no invoke instruction).
      // We assume the arguments are available as parameters in the method body.
      if (paramPos >= 0) {
        return getArgumentValueNumber(this.getNode(), paramPos);
      }
    }

    if (this.getNode().getMethod().getName().toString().equals("read_data")) {
      // For read_data nodes, we don't have explicit arguments in the IR.
      // Returning MAX_VALUE acts as a sentinel to bypass the "missing argument" check below
      // and allows getDTypes/getShapes to proceed to getArgumentPointsToSet,
      // which correctly delegates argument resolution to the caller (do).
      return Integer.MAX_VALUE;
    }

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
    // 1. Try to resolve the call directly from the definition of the value.
    // This works if we are analyzing the code where the function was called (the caller).
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      return call.getKeywords().contains(paramName);
    }

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

    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int numKeywords = call.getKeywords() != null ? call.getKeywords().size() : 0;
      ret.add(call.getNumberOfUses() - 1 - numKeywords);
      return ret;
    }

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
          int numKeywords =
              pyCallInstr.getKeywords() != null ? pyCallInstr.getKeywords().size() : 0;
          int numberOfPositionalParameters =
              pyCallInstr.getNumberOfUses()
                  - 1
                  - numKeywords; // Exclude the function name and keywords.

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

  /**
   * Creates a manual TensorGenerator for synthetic operations where standard points-to analysis
   * fails (e.g. UnimplementedError due to implicit pointer keys for allocations in synthetic do()
   * methods).
   */
  private static TensorGenerator createManualGenerator(
      CGNode node, PropagationCallGraphBuilder builder) {
    TypeReference type = node.getMethod().getDeclaringClass().getReference();
    if (type.equals(TensorFlowTypes.ONES.getDeclaringClass())) {
      return new Ones(node);
    } else if (type.equals(TensorFlowTypes.SPARSE_EYE.getDeclaringClass())) {
      return new SparseEye(node);
    } else if (type.equals(TensorFlowTypes.EYE.getDeclaringClass())) {
      return new Eye(node);
    } else if (type.equals(TensorFlowTypes.MATMUL.getDeclaringClass())) {
      return new MatMul(node);
    } else if (type.equals(CONSTANT.getDeclaringClass())) {
      return new TensorGenerator(node) {
        @Override
        public String toString() {
          return "Manual Constant Generator";
        }

        @Override
        protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
          // If dtype is not provided, infer from the value (arg 0).
          return this.getDTypes(builder, this.getArgumentValueNumber(0));
        }

        @Override
        protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
          // If the shape argument is not provided, we infer the shape from the value (arg 0).
          return this.getShapes(builder, this.getArgumentValueNumber(0));
        }

        @Override
        protected int getShapeParameterPosition() {
          return UNDEFINED_PARAMETER_POSITION;
        }

        @Override
        protected String getShapeParameterName() {
          return "shape";
        }

        @Override
        protected int getDTypeParameterPosition() {
          return 1;
        }

        @Override
        protected String getDTypeParameterName() {
          return "dtype";
        }
      };
    } else if (type.equals(PLACEHOLDER.getDeclaringClass())) {
      return new Placeholder(node);
    }
    return null;
  }
}
