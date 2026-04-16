package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.ml.client.PythonTensorAnalysisEngine.TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.CONSTANT_OP_CONSTANT;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_CHOOSE_FROM_DATASETS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DATASET_SAMPLE_FROM_DATASETS_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.INT32;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.STRING;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.FIELD_REFERENCE_TO_DTYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.PLACEHOLDER;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TENSORFLOW_TYPE;
import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.TYPE_REFERENCE_TO_SIGNATURE;
import static com.ibm.wala.cast.python.types.PythonTypes.DO_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.findDefinition;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.getFunction;
import static com.ibm.wala.cast.python.util.Util.getReceiverValueNumber;
import static com.ibm.wala.cast.python.util.Util.sanitize;
import static com.ibm.wala.core.util.strings.Atom.findOrCreateAsciiAtom;
import static com.ibm.wala.ipa.callgraph.propagation.cfa.CallStringContextSelector.CALL_STRING;
import static java.util.Arrays.asList;
import static java.util.Collections.emptyList;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.CompoundDim;
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
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.Collections;
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
 * <h2>Lattice conventions for shapes and dtypes</h2>
 *
 * Subclasses <strong>must</strong> follow these conventions so that downstream consumers can
 * distinguish "unknown tensor" (⊤) from "not a tensor" (⊥):
 *
 * <h3>Shapes — {@link #getDefaultShapes(PropagationCallGraphBuilder)}</h3>
 *
 * <ul>
 *   <li>{@code null} — ⊤, the generator produces a tensor but its shape cannot be determined.
 *   <li>empty set ({@code Collections.emptySet()}) — ⊥, the variable is provably not a tensor.
 *   <li>non-empty set — the set of concrete shapes the tensor may take.
 * </ul>
 *
 * <p>Within a single shape, use {@link TensorType.SymbolicDim}{@code ("?")} for a
 * known-rank-but-unknown-size dimension (e.g., a dynamic batch size). A {@code null} shape list
 * means even the rank is unknown.
 *
 * <h3>Dtypes — {@link #getDefaultDTypes(PropagationCallGraphBuilder)}</h3>
 *
 * <ul>
 *   <li>{@code EnumSet.of(DType.UNKNOWN)} — ⊤, the generator produces a tensor but its dtype cannot
 *       be determined. Never return a bare empty set for the "unknown" case.
 *   <li>empty set — ⊥, the variable is provably not a tensor.
 *   <li>non-empty set of concrete {@link DType}s — the set of possible dtypes.
 * </ul>
 *
 * <h3>Tensor types — {@link #getTensorTypes(PropagationCallGraphBuilder)}</h3>
 *
 * Shapes and dtypes are orthogonal. When the shape is unknown but the dtype is known, {@link
 * #getTensorTypes(PropagationCallGraphBuilder)} emits {@code TensorType} instances with {@code
 * null} dims so dtype information is preserved. {@link TensorType} is null-dims-safe; subclasses
 * consuming {@link TensorType}s must also be.
 *
 * <p>When adding a new {@link TensorGenerator} subclass, audit every final-fallback return in
 * {@code getDefaultShapes} and {@code getDefaultDTypes} against the table above — the most common
 * mistake is returning {@code Collections.emptySet()} when the intended meaning is "unknown."
 *
 * <p>TODO: Revisit caching of shapes and dtypes.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public abstract class TensorGenerator {

  protected static final int UNDEFINED_PARAMETER_POSITION = -1;

  protected static final Logger LOGGER = Logger.getLogger(TensorGenerator.class.getName());

  /** The source of the tensor, represented by a points-to set variable. */
  protected PointsToSetVariable source;

  /**
   * The call graph node representing the "manual" generator. A generator is considered manual when
   * it is instantiated directly from a Call Graph Node ({@link CGNode}) rather than a points-to set
   * variable. This fallback mechanism is used when WALA's pointer analysis cannot construct a
   * trackable points-to set for the tensor's allocation site (often due to limitations dealing with
   * implicit allocations in synthetic model methods). In such cases, the generator analyzes the
   * instructions directly within this node's IR.
   */
  protected CGNode manualNode;

  /**
   * Constructs a new tensor generator based on a standard points-to set source.
   *
   * @param source The points-to set variable representing the source of the tensor.
   */
  public TensorGenerator(PointsToSetVariable source) {
    this.source = source;
  }

  /**
   * Constructs a new "manual" tensor generator based directly on a call graph node.
   *
   * @param node The call graph node representing the operation that generates the tensor. Used as a
   *     fallback when standard pointer analysis fails to provide a trackable source.
   */
  public TensorGenerator(CGNode node) {
    this.manualNode = node;
  }

  /**
   * Returns a set of possible {@link TensorType}s that this generator can produce, or {@code null}
   * if this generator is known to produce a tensor but its shape cannot be determined (unknown /
   * top). An empty set means the variable has no possible tensor type (i.e., it is not a tensor).
   *
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return A set of possible {@link TensorType}s, or {@code null} if the shape is unknown.
   */
  public Set<TensorType> getTensorTypes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> shapes = this.getShapes(builder);
    Set<DType> dTypes = this.getDTypes(builder);

    // If we have no dtype info at all, fall back to signaling "unknown tensor" when shapes are
    // also unknown, otherwise produce an empty set (⊥, not a tensor).
    if (dTypes.isEmpty()) {
      return shapes == null ? null : HashSetFactory.make();
    }

    Set<TensorType> ret = HashSetFactory.make();

    if (shapes == null) {
      // Shape is unknown (⊤), but dtype info may still be available. Emit TensorTypes with null
      // dims so the dtype information is preserved.
      for (DType dtype : dTypes) ret.add(new TensorType(dtype.name().toLowerCase(), null));
    } else {
      // Create a tensor type for each possible shape and dtype combination.
      for (List<Dimension<?>> dimensionList : shapes)
        for (DType dtype : dTypes)
          ret.add(new TensorType(dtype.name().toLowerCase(), dimensionList));
    }

    LOGGER.info("Generator " + this.getClass().getSimpleName() + " produced types: " + ret);

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
        // correspondences to the set of possible dimensions for that index.
        if (objectCatalogPointsToSet.isEmpty()) {
          ret.add(Collections.emptyList());
          continue;
        }
        @SuppressWarnings({"unchecked", "rawtypes"})
        Set<Dimension<?>>[] possibleDimensions = new Set[objectCatalogPointsToSet.size()];

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
          Set<Dimension<?>> tensorDimensions = HashSetFactory.make();

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
            } else if (instanceFieldIK instanceof AllocationSiteInNode) {
              AllocationSiteInNode innerAsin = (AllocationSiteInNode) instanceFieldIK;
              TypeReference innerReference = innerAsin.getConcreteType().getReference();

              if (innerReference.equals(tuple)
                  || innerReference.equals(list)
                  || innerReference.equals(TensorFlowTypes.TENSOR_SPEC)
                  || innerReference.equals(TensorFlowTypes.RAGGED_TENSOR_SPEC)) {
                // Nested tuple/list or Spec. Recurse.
                Set<List<Dimension<?>>> nestedShapes =
                    this.getShapesFromShapeArgument(
                        builder, Collections.singleton(instanceFieldIK));

                for (List<Dimension<?>> nestedShape : nestedShapes)
                  tensorDimensions.add(new CompoundDim(nestedShape));
              } else
                throw new IllegalStateException(
                    "Expected a constant key or nested structure for instance field: "
                        + pointerKeyForInstanceField
                        + ", but got: "
                        + instanceFieldIK
                        + ".");
            } else
              throw new IllegalStateException(
                  "Expected a constant key or nested structure for instance field: "
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
          for (Dimension<?> iDim : possibleDimensions[i]) {
            @SuppressWarnings({"unchecked", "rawtypes"})
            Dimension<?>[] dimensions = new Dimension[possibleDimensions.length];

            dimensions[i] = iDim;

            for (int j = 0; j < possibleDimensions.length; j++)
              if (i != j) for (Dimension<?> jDim : possibleDimensions[j]) dimensions[j] = jDim;

            ret.add(asList(dimensions));
          }
      } else if (reference.equals(CONSTANT_OP_CONSTANT)) {
        // We have a constant tensor. We recurse into its value field.
        IField valueField =
            builder.getClassHierarchy().resolveField(TensorFlowTypes.CONSTANT_VALUE);
        PointerKey valuePK = builder.getPointerKeyForInstanceField(instanceKey, valueField);
        OrdinalSet<InstanceKey> valuePts = pointerAnalysis.getPointsToSet(valuePK);
        ret.addAll(this.getShapesFromShapeArgument(builder, valuePts));
      } else if (reference.equals(TensorFlowTypes.TENSOR_SPEC)
          || reference.equals(TensorFlowTypes.RAGGED_TENSOR_SPEC)) {
        // We have a TensorSpec or RaggedTensorSpec. These objects carry shape and dtype
        // information in their fields. We extract the 'shape' field and recurse to
        // parse the actual shape structure (usually a tuple or list of integers).
        IField shapeField =
            builder
                .getClassHierarchy()
                .resolveField(
                    reference.equals(TensorFlowTypes.TENSOR_SPEC)
                        ? TensorFlowTypes.SPEC_SHAPE
                        : TensorFlowTypes.RAGGED_SPEC_SHAPE);
        PointerKey shapePK = builder.getPointerKeyForInstanceField(instanceKey, shapeField);
        OrdinalSet<InstanceKey> shapePts = pointerAnalysis.getPointsToSet(shapePK);
        ret.addAll(this.getShapesFromShapeArgument(builder, shapePts));
      } else
        throw new IllegalStateException(
            "Expected a " + list + " or " + tuple + " for the shape, but got: " + reference + ".");
    }

    return ret;
  }

  protected static Integer getFieldIndex(ConstantKey<?> constantKey) {
    Object constantKeyValue = constantKey.getValue();

    if (constantKeyValue instanceof Integer) return (Integer) constantKeyValue;
    else if (constantKeyValue instanceof String) return Integer.parseInt((String) constantKeyValue);

    return null;
  }

  /**
   * Returns the default shapes when no explicit shape argument is provided. Implementations should
   * return {@code null} when the generator is known to produce a tensor but its shape cannot be
   * determined (unknown / ⊤). An empty set should be returned only when the variable is provably
   * not a tensor (⊥). A non-empty set carries concrete shape information.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The default shapes, or {@code null} if the shape is unknown.
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
    return getShapes(builder, this.getNode(), valueNumber);
  }

  /**
   * Returns the possible shapes of the tensor represented by the given value number in the
   * specified node. This method uses a multi-staged approach, falling back to interprocedural
   * generator-based tracing if standard points-to analysis fails.
   */
  protected Set<List<Dimension<?>>> getShapes(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    PointerKey valuePK = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, valueNumber);
    OrdinalSet<InstanceKey> valuePointsToSet = pointerAnalysis.getPointsToSet(valuePK);

    LOGGER.fine(
        () ->
            "getShapes(node, vn): node="
                + node
                + ", vn="
                + valueNumber
                + ", ptsEmpty="
                + valuePointsToSet.isEmpty());

    if (!valuePointsToSet.isEmpty()) {
      Set<List<Dimension<?>>> shapes = this.getShapesOfValue(builder, valuePointsToSet);
      if (shapes == null || !shapes.isEmpty()) {
        return shapes;
      }
    }

    // points-to set is empty. Try to find a generator for this variable.
    boolean implicit = builder.getPropagationSystem().isImplicit(valuePK);
    PointsToSetVariable var = null;
    if (!implicit) {
      var = builder.getPropagationSystem().findOrCreatePointsToSet(valuePK);
    }

    if (var != null) {
      try {
        TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
        // Recurse into the generator as long as it's not the *same* generator (identical source)
        // as `this`. Previously this check compared classes, which prevented an
        // `ElementWiseOperation` from recursing into a nested `ElementWiseOperation` on a
        // different operand value number — exactly the case for `(x - k1) / k2` chains.
        if (generator != null && !generator.getClass().equals(this.getClass())) {
          LOGGER.fine(
              () ->
                  "getShapes(node, vn): recovering via factory generator "
                      + generator.getClass().getSimpleName()
                      + " for vn="
                      + valueNumber);
          return generator.getShapes(builder);
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(
            Level.FINE,
            "getShapes(node, vn): factory IAE for vn=" + valueNumber + ": " + e.getMessage(),
            e);
      }
    }

    // No direct generator. Try tracing the definition or parameters.
    SSAInstruction def = node.getDU().getDef(valueNumber);
    if (def == null) {
      // It's a parameter. Trace back to call sites.
      int paramPos = -1;
      for (int i = 0; i < node.getIR().getNumberOfParameters(); i++) {
        if (node.getIR().getParameter(i) == valueNumber) {
          paramPos = node.getMethod().isStatic() ? i : i - 1;
          break;
        }
      }

      if (paramPos >= -1) { // -1 is 'self'
        Set<List<Dimension<?>>> combinedRet = HashSetFactory.make();
        CallString cs = (CallString) node.getContext().get(CALL_STRING);
        if (cs != null) {
          for (int i = 0; i < cs.getCallSiteRefs().length; i++) {
            CallSiteReference site = cs.getCallSiteRefs()[i];
            IMethod callerMethod = cs.getMethods()[i];
            for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(node); it.hasNext(); ) {
              CGNode caller = it.next();
              if (caller.getMethod().equals(callerMethod)) {
                for (SSAAbstractInvokeInstruction call : caller.getIR().getCalls(site)) {
                  int argVn = -1;
                  if (paramPos == -1) { // self
                    argVn = call.getUse(0);
                  } else if (call instanceof PythonInvokeInstruction) {
                    // Try to find the argument index. This is simplified.
                    if (paramPos + 1 < call.getNumberOfUses()) {
                      argVn = call.getUse(paramPos + 1);
                    }
                  } else if (paramPos < call.getNumberOfUses()) {
                    argVn = call.getUse(paramPos);
                  }

                  if (argVn != -1) {
                    Set<List<Dimension<?>>> argShapes = this.getShapes(builder, caller, argVn);
                    if (argShapes != null) combinedRet.addAll(argShapes);
                  }
                }
              }
            }
          }
          if (!combinedRet.isEmpty()) return combinedRet;
        }
      }
    }

    throw new IllegalArgumentException(
        "Empty points-to set and could not trace properties for value number: "
            + valueNumber
            + " in: "
            + node
            + ".");
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
        } else if (reference.equals(TensorFlowTypes.D_TYPE)) {
          LOGGER.fine("Ignoring DType: " + asin);
        } else if (reference.equals(TensorFlowTypes.FEATURE)) {
          LOGGER.fine("Ignoring feature: " + asin);
        } else {
          // Assume the value is a tensor and attempt to find the generator that created it
          // to ask for its shape.
          LOGGER.fine(
              "Encountered "
                  + reference.getName()
                  + ". Attempting to retrieve shape from producer.");
          Set<List<Dimension<?>>> fromTensor = this.getShapesFromTensor(builder, asin);
          if (fromTensor != null) ret.addAll(fromTensor);
        }
      } else if (getAllocationSiteInNode(valueIK) != null) {
        // Unwrap ScopeMappingInstanceKey or similar wrapping keys
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);

        // Instead of forcing a points-to set, try to get the generator for this allocation site
        PointerKey pk =
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getPointerKeyForLocal(asin.getNode(), asin.getSite().getProgramCounter());
        PointsToSetVariable var = null;
        if (!builder.getPropagationSystem().isImplicit(pk)) {
          var = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
        }
        try {
          TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
          if (generator != null && !generator.getClass().equals(this.getClass())) {
            Set<List<Dimension<?>>> generatorShapes = generator.getShapes(builder);
            if (generatorShapes != null) ret.addAll(generatorShapes);
          }
        } catch (IllegalArgumentException e) {
          // Not a recognized generator.
          LOGGER.log(Level.FINE, "No generator found for variable: " + var, e);
        }
      } else {
        throw new IllegalStateException("Unknown value type: " + valueIK.getClass() + ".");
      }

    return ret;
  }

  /**
   * Retrieves the shapes of a tensor that is the result of another TensorFlow operation (the
   * "producer").
   *
   * <p>This method traces the tensor back to its allocation site to identify the operation that
   * created it. It handles two main scenarios:
   *
   * <ol>
   *   <li><b>Direct Allocation in `do`:</b> The tensor is allocated directly within the `do` method
   *       of the operation. It attempts to find the definition of the tensor and trace it back to a
   *       {@link PointsToSetVariable}. If successful, it delegates to the generator for that
   *       source. If points-to analysis fails (e.g., due to implicit pointer keys), it attempts to
   *       create a manual generator using {@link #createManualGenerator(CGNode,
   *       PropagationCallGraphBuilder)}.
   *   <li><b>Helper Method (`read_data`):</b> The tensor is allocated in a helper method (like
   *       `read_data`) called by `do`. It identifies the call site in `do` that invoked the helper
   *       and recursively delegates to the generator for the result of that call.
   * </ol>
   *
   * @param builder the propagation call graph builder
   * @param asin the allocation site of the tensor
   * @return a set of possible shapes for the tensor
   */
  private Set<List<Dimension<?>>> getShapesFromTensor(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    CGNode readDataNode = asin.getNode();

    // Support allocations directly in 'do' methods (preferred for 1-CFA context separation).
    if (readDataNode.getMethod().getName().toString().equals(DO_METHOD_NAME)) {
      int def = findDefinition(readDataNode, asin);
      if (def != -1) {
        PointerKey defKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(readDataNode, def);
        PointsToSetVariable defSource = null;
        if (!builder.getPropagationSystem().isImplicit(defKey)) {
          defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
        }

        TensorGenerator generator = createManualGenerator(readDataNode, builder);

        if (generator != null) {
          // Avoid infinite recursion for manual generators
          if (this.manualNode != null && this.manualNode.equals(readDataNode)) {
            return ret;
          }
          LOGGER.fine("Delegating shape inference to: " + generator);
          Set<List<Dimension<?>>> delegatedShapes = generator.getShapes(builder);
          if (delegatedShapes != null) ret.addAll(delegatedShapes);
        } else if (defSource != null) {
          // Avoid infinite recursion if the current generator is for the same source.
          if (this.getSource() != null && this.getSource().equals(defSource)) {
            return ret;
          }
          try {
            generator = TensorGeneratorFactory.getGenerator(defSource, builder);
          } catch (IllegalArgumentException e) {
            // Factory couldn't resolve — treat as "no generator" and skip. See wala/ML#363.
            LOGGER.log(Level.FINE, "Delegating shape inference: factory IAE for " + defSource, e);
            generator = null;
          }
          if (generator != null) {
            LOGGER.fine("Delegating shape inference to: " + generator);
            Set<List<Dimension<?>>> delegatedShapes = generator.getShapes(builder);
            if (delegatedShapes != null) ret.addAll(delegatedShapes);
          }
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
              PointsToSetVariable defSource = null;
              if (!builder.getPropagationSystem().isImplicit(defKey)) {
                defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
              }

              // Try to create a manual generator for the caller (doNode) first.
              TensorGenerator generator = createManualGenerator(doNode, builder);
              if (generator == null) {
                try {
                  generator = TensorGeneratorFactory.getGenerator(defSource, builder);
                } catch (IllegalArgumentException e) {
                  // Factory couldn't resolve — treat as "no generator". See wala/ML#363.
                  LOGGER.log(
                      Level.FINE, "Delegating shape inference: factory IAE for " + defSource, e);
                  generator = null;
                }
              }

              if (generator != null) {
                LOGGER.fine("Delegating shape inference to: " + generator);
                Set<List<Dimension<?>>> delegatedShapes = generator.getShapes(builder);
                if (delegatedShapes != null) ret.addAll(delegatedShapes);
              }
            }
          }
        }
      }
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
      AllocationSiteInNode asin = getAllocationSiteInNode(instanceKey);
      if (asin == null && !(instanceKey instanceof ConstantKey)) continue;
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
                  .getInstanceKeyForAllocation(
                      importNode, NewSiteReference.make(0, TENSORFLOW_TYPE));

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
      } else if (typeReference.equals(TensorFlowTypes.CONSTANT_OP_CONSTANT)) {
        // We have a constant tensor. We extract its 'dtype' field and recurse to
        // resolve the actual DType value.
        IField valueField =
            builder.getClassHierarchy().resolveField(TensorFlowTypes.CONSTANT_DTYPE);
        PointerKey valuePK = builder.getPointerKeyForInstanceField(instanceKey, valueField);
        OrdinalSet<InstanceKey> valuePts = pointerAnalysis.getPointsToSet(valuePK);
        if (valuePts != null && !valuePts.isEmpty()) {
          ret.addAll(this.getDTypesFromDTypeArgument(builder, valuePts));
        }
      } else if (typeReference.equals(TensorFlowTypes.TENSOR_SPEC)
          || typeReference.equals(TensorFlowTypes.RAGGED_TENSOR_SPEC)) {
        // We have a TensorSpec or RaggedTensorSpec. We extract the 'dtype' field and recurse to
        // resolve the actual DType value (usually a tf.DType instance).
        IField dtypeField =
            builder
                .getClassHierarchy()
                .resolveField(
                    typeReference.equals(TensorFlowTypes.TENSOR_SPEC)
                        ? TensorFlowTypes.SPEC_DTYPE
                        : TensorFlowTypes.RAGGED_SPEC_DTYPE);
        PointerKey dtypePK = builder.getPointerKeyForInstanceField(instanceKey, dtypeField);
        OrdinalSet<InstanceKey> dtypePts = pointerAnalysis.getPointsToSet(dtypePK);
        if (dtypePts != null && !dtypePts.isEmpty()) {
          ret.addAll(this.getDTypesFromDTypeArgument(builder, dtypePts));
        }
      } else if (typeReference.equals(tuple) || typeReference.equals(list)) {
        OrdinalSet<InstanceKey> objectCatalogPointsToSet =
            pointerAnalysis.getPointsToSet(
                ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                    .getPointerKeyForObjectCatalog(asin));

        for (InstanceKey catalogIK : objectCatalogPointsToSet) {
          ConstantKey<?> constantKey = (ConstantKey<?>) catalogIK;
          Integer fieldIndex = getFieldIndex(constantKey);

          FieldReference subscript =
              FieldReference.findOrCreate(Root, findOrCreateAsciiAtom(fieldIndex.toString()), Root);

          IField f = builder.getClassHierarchy().resolveField(subscript);
          if (f != null) {
            PointerKey pk = builder.getPointerKeyForInstanceField(asin, f);
            OrdinalSet<InstanceKey> fieldPts = pointerAnalysis.getPointsToSet(pk);
            ret.addAll(this.getDTypesFromDTypeArgument(builder, fieldPts));
          }
        }
      } else if (instanceKey instanceof ConstantKey
          && ((ConstantKey<?>) instanceKey).getValue() instanceof String) {
        String value = (String) ((ConstantKey<?>) instanceKey).getValue();
        DType dtype = null;

        try {
          dtype = DType.valueOf(value.toUpperCase()); // Validate the dtype string.
        } catch (IllegalArgumentException | NullPointerException e) {
          if (value.equals("float")) {
            dtype = FLOAT32;
          } else {
            throw new IllegalStateException("Unknown dtype string: " + value + ".", e);
          }
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
   * <p>Implementations should return {@link DType#UNKNOWN} (i.e., {@code EnumSet.of(DType.UNKNOWN)}
   * or {@code Set.of(DType.UNKNOWN)}) to indicate that the dtype cannot be determined. An empty set
   * means the variable is not a tensor at all (⊥). A set of concrete dtypes means the dtype is
   * known.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible dtypes of the tensor returned by this generator when an explicit
   *     dtype isn't provided as an argument, or a set containing {@link DType#UNKNOWN} if the dtype
   *     cannot be determined.
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
    return getDTypes(builder, this.getNode(), valueNumber);
  }

  /**
   * Returns the possible dtypes of the tensor represented by the given value number in the
   * specified node.
   */
  protected Set<DType> getDTypes(
      PropagationCallGraphBuilder builder, CGNode node, int valueNumber) {
    PointerAnalysis<InstanceKey> pointerAnalysis = builder.getPointerAnalysis();
    PointerKey valuePK = pointerAnalysis.getHeapModel().getPointerKeyForLocal(node, valueNumber);
    OrdinalSet<InstanceKey> valuePointsToSet = pointerAnalysis.getPointsToSet(valuePK);

    if (valuePointsToSet != null && !valuePointsToSet.isEmpty()) {
      Set<DType> dtypes = this.getDTypesOfValue(builder, valuePointsToSet);
      if (!dtypes.isEmpty()) {
        return dtypes;
      }
    }

    // points-to set is empty or yielded no dtypes. Try to find a generator for this variable.
    PointsToSetVariable var = null;
    if (!builder.getPropagationSystem().isImplicit(valuePK)) {
      var = builder.getPropagationSystem().findOrCreatePointsToSet(valuePK);
    }

    if (var != null) {
      try {
        TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
        // See `getShapes(builder, CGNode, int)` — we compare by source, not class, so two
        // different `ElementWiseOperation` generators for different operand value numbers can
        // still recurse into each other.
        if (generator != null && !generator.getClass().equals(this.getClass())) {
          return generator.getDTypes(builder);
        }
      } catch (IllegalArgumentException e) {
        LOGGER.log(Level.FINE, "Not a recognized generator: " + var, e);
      }
    }

    // No direct generator. Try tracing the definition or parameters.
    SSAInstruction def = node.getDU().getDef(valueNumber);
    if (def == null) {
      // It's a parameter. Trace back to call sites.
      int paramPos = -1;
      for (int i = 0; i < node.getIR().getNumberOfParameters(); i++) {
        if (node.getIR().getParameter(i) == valueNumber) {
          paramPos = node.getMethod().isStatic() ? i : i - 1;
          break;
        }
      }

      if (paramPos >= -1) {
        Set<DType> combinedRet = EnumSet.noneOf(DType.class);
        CallString cs = (CallString) node.getContext().get(CALL_STRING);
        if (cs != null) {
          for (int i = 0; i < cs.getCallSiteRefs().length; i++) {
            CallSiteReference site = cs.getCallSiteRefs()[i];
            IMethod callerMethod = cs.getMethods()[i];
            for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(node); it.hasNext(); ) {
              CGNode caller = it.next();
              if (caller.getMethod().equals(callerMethod)) {
                for (SSAAbstractInvokeInstruction call : caller.getIR().getCalls(site)) {
                  int argVn = -1;
                  if (paramPos == -1) { // self
                    argVn = call.getUse(0);
                  } else if (call instanceof PythonInvokeInstruction) {
                    if (paramPos + 1 < call.getNumberOfUses()) {
                      argVn = call.getUse(paramPos + 1);
                    }
                  } else if (paramPos < call.getNumberOfUses()) {
                    argVn = call.getUse(paramPos);
                  }

                  if (argVn != -1) {
                    combinedRet.addAll(this.getDTypes(builder, caller, argVn));
                  }
                }
              }
            }
          }
          if (!combinedRet.isEmpty()) return combinedRet;
        }
      }
    }

    throw new IllegalArgumentException(
        "Empty points-to set and could not trace properties for value number: "
            + valueNumber
            + " in: "
            + node
            + ".");
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
        } else if (reference.equals(TensorFlowTypes.FEATURE)) {
          // Ignore features.
          LOGGER.fine("Ignoring feature: " + asin);
        } else if (reference.equals(TensorFlowTypes.D_TYPE)) {
          // Ignore DTypes.
          LOGGER.fine("Ignoring DType: " + asin);
        } else {
          // Assume the value is a tensor and attempt to find the generator that created it
          // to ask for its dtype.
          LOGGER.fine(
              "Encountered "
                  + reference.getName()
                  + ". Attempting to retrieve dtype from producer.");
          ret.addAll(this.getDTypesFromTensor(builder, asin));
        }
      } else if (getAllocationSiteInNode(valueIK) != null) {
        // Unwrap ScopeMappingInstanceKey or similar wrapping keys
        AllocationSiteInNode asin = getAllocationSiteInNode(valueIK);

        // Instead of forcing a points-to set, try to get the generator for this allocation site
        PointerKey pk =
            builder
                .getPointerAnalysis()
                .getHeapModel()
                .getPointerKeyForLocal(asin.getNode(), asin.getSite().getProgramCounter());
        PointsToSetVariable var = null;
        if (!builder.getPropagationSystem().isImplicit(pk)) {
          var = builder.getPropagationSystem().findOrCreatePointsToSet(pk);
        }
        try {
          TensorGenerator generator = TensorGeneratorFactory.getGenerator(var, builder);
          if (generator != null && !generator.getClass().equals(this.getClass())) {
            ret.addAll(generator.getDTypes(builder));
          }
        } catch (IllegalArgumentException e) {
          // Factory couldn't resolve — skip this instance. See wala/ML#363.
          LOGGER.log(Level.FINE, "getDTypesOfValue: factory IAE for " + var, e);
        }
      } else {
        throw new IllegalStateException("Unknown value type: " + valueIK.getClass() + ".");
      }
    }

    return ret;
  }

  private Set<DType> getDTypesFromTensor(
      PropagationCallGraphBuilder builder, AllocationSiteInNode asin) {
    Set<DType> ret = EnumSet.noneOf(DType.class);
    CGNode readDataNode = asin.getNode();

    // Support allocations directly in 'do' methods (preferred for 1-CFA context separation).
    if (readDataNode.getMethod().getName().toString().equals(DO_METHOD_NAME)) {
      int def = findDefinition(readDataNode, asin);
      if (def != -1) {
        PointerKey defKey =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(readDataNode, def);
        PointsToSetVariable defSource = null;
        if (!builder.getPropagationSystem().isImplicit(defKey)) {
          defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
        }

        TensorGenerator generator = createManualGenerator(readDataNode, builder);

        if (generator != null) {
          if (this.manualNode != null && this.manualNode.equals(readDataNode)) {
            return ret;
          }
          LOGGER.fine("Delegating dtype inference to: " + generator);
          ret.addAll(generator.getDTypes(builder));
        } else if (defSource != null) {
          if (this.getSource() != null && this.getSource().equals(defSource)) {
            return ret;
          }

          try {
            generator = TensorGeneratorFactory.getGenerator(defSource, builder);
          } catch (IllegalArgumentException e) {
            // Factory couldn't resolve — treat as "no generator". See wala/ML#363.
            LOGGER.log(Level.FINE, "Delegating dtype inference: factory IAE for " + defSource, e);
            generator = null;
          }
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
              PointsToSetVariable defSource = null;
              if (!builder.getPropagationSystem().isImplicit(defKey)) {
                defSource = builder.getPropagationSystem().findOrCreatePointsToSet(defKey);
              }

              // Try to create a manual generator for the caller (doNode) first.
              TensorGenerator generator = createManualGenerator(doNode, builder);
              if (generator == null) {
                try {
                  generator = TensorGeneratorFactory.getGenerator(defSource, builder);
                } catch (IllegalArgumentException e) {
                  // Factory couldn't resolve — treat as "no generator". See wala/ML#363.
                  LOGGER.log(
                      Level.FINE, "Delegating dtype inference: factory IAE for " + defSource, e);
                  generator = null;
                }
              }

              if (generator != null) {
                LOGGER.fine("Delegating dtype inference to: " + generator);
                ret.addAll(generator.getDTypes(builder));
              }
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

  /**
   * Two generators are equal iff they are of the same concrete class AND share the same identity on
   * both the source and the manual-node axes. For factory-constructed generators the source is
   * populated; for manually-constructed generators the manual node is. Both fields participate in
   * the comparison so that a source-based generator and a manual-node-based generator are not
   * accidentally considered equal when they happen to share one half of the identity but not the
   * other.
   *
   * <p>Used by {@link #getShapes(PropagationCallGraphBuilder, CGNode, int)} and its dtype
   * counterpart to avoid infinite recursion when dispatching back through {@link
   * TensorGeneratorFactory#getGenerator}. A coarser class-only equality would incorrectly conflate
   * two different generators on different value numbers or different call graph nodes — blocking
   * legitimate recursion such as nested {@link ElementWiseOperation} binop chains.
   */
  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (this.getClass() != obj.getClass()) return false;
    TensorGenerator other = (TensorGenerator) obj;
    return this.source == other.source && this.manualNode == other.manualNode;
  }

  @Override
  public int hashCode() {
    int result = this.getClass().hashCode();
    result = 31 * result + System.identityHashCode(this.source);
    result = 31 * result + System.identityHashCode(this.manualNode);
    return result;
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
    String signature = TYPE_REFERENCE_TO_SIGNATURE.get(function);
    if (signature == null) {
      LOGGER.warning(
          "Unmapped TensorGenerator for function: "
              + function
              + ". Either add an entry to TYPE_REFERENCE_TO_SIGNATURE or fix the dispatch"
              + " that created a generator for a non-TF function.");
      return "<unmapped:" + function + ">";
    }
    return signature;
  }

  protected static final int RECEIVER_PARAMETER_POSITION = -2;

  protected static final String SELF = "self";

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

  protected SSABinaryOpInstruction getBinaryOpInstruction() {
    if (this.source == null) {
      return null;
    }
    if (this.getSource().getPointerKey() instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) this.getSource().getPointerKey();
      if (lpk.getNode().equals(this.getNode())) {
        int vn = lpk.getValueNumber();
        if (vn > 0) {
          SSAInstruction def = this.getNode().getDU().getDef(vn);
          if (def instanceof SSABinaryOpInstruction) {
            return (SSABinaryOpInstruction) def;
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

  /**
   * Resolves shapes for the argument at the given parameter position by walking the {@link
   * CallString} context to find each caller's invocation site, then delegating to {@link
   * #getShapes(PropagationCallGraphBuilder, CGNode, int)} with the caller's value number. This is a
   * fallback path used when {@link #getArgumentPointsToSet(PropagationCallGraphBuilder, int,
   * String)} returns an empty set because the argument's points-to set is empty — commonly the case
   * when the argument is the result of a Python binary op on tensors, for which WALA does not
   * allocate a trackable target. The caller-side recursion into {@link #getShapes(
   * PropagationCallGraphBuilder, CGNode, int)} picks up the {@link ElementWiseOperation} (or
   * similar) generator via {@link TensorGeneratorFactory}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @param paramPos The 0-based index of the positional parameter (excluding {@code self} for
   *     instance methods).
   * @param paramName The name of the keyword parameter, or {@code null}.
   * @return The union of shapes resolved from each caller, or {@code null} if no caller could be
   *     resolved.
   */
  protected Set<List<Dimension<?>>> getArgumentShapesViaCallers(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    CallString cs = (CallString) this.getNode().getContext().get(CALL_STRING);
    LOGGER.fine(() -> "getArgumentShapesViaCallers: node=" + this.getNode() + ", cs=" + cs);
    if (cs == null) return null;
    Set<List<Dimension<?>>> combined = null;
    for (int i = 0; i < cs.getCallSiteRefs().length; i++) {
      CallSiteReference siteRef = cs.getCallSiteRefs()[i];
      IMethod callerMethod = cs.getMethods()[i];
      for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(this.getNode());
          it.hasNext(); ) {
        CGNode caller = it.next();
        if (!caller.getMethod().equals(callerMethod)) continue;
        for (SSAAbstractInvokeInstruction callInstr : caller.getIR().getCalls(siteRef)) {
          if (!(callInstr instanceof PythonInvokeInstruction)) continue;
          PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callInstr;
          int argVn = -1;
          if (paramName != null) argVn = pyCall.getUse(paramName);
          if (argVn == -1 && paramPos >= 0) {
            int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
            if (paramPos < numPosParams) argVn = pyCall.getUse(paramPos + 1);
          }
          if (argVn <= 0) continue;
          final int finalArgVn = argVn;
          final CGNode finalCaller = caller;
          try {
            Set<List<Dimension<?>>> argShapes = this.getShapes(builder, caller, argVn);
            LOGGER.fine(
                () -> "getArgumentShapesViaCallers: argVn=" + finalArgVn + " shapes=" + argShapes);
            if (argShapes != null && !argShapes.isEmpty()) {
              if (combined == null) combined = HashSetFactory.make();
              combined.addAll(argShapes);
            }
          } catch (IllegalArgumentException e) {
            LOGGER.log(
                Level.FINE,
                "getArgumentShapesViaCallers: IAE for argVn="
                    + finalArgVn
                    + " in caller="
                    + finalCaller,
                e);
          }
        }
      }
    }
    return combined;
  }

  /**
   * Dtype counterpart of {@link #getArgumentShapesViaCallers(PropagationCallGraphBuilder, int,
   * String)}. See that method for the rationale; the behaviour is the same but returns a set of
   * {@link DType}s resolved via {@link #getDTypes(PropagationCallGraphBuilder, CGNode, int)}.
   */
  protected Set<DType> getArgumentDTypesViaCallers(
      PropagationCallGraphBuilder builder, int paramPos, String paramName) {
    CallString cs = (CallString) this.getNode().getContext().get(CALL_STRING);
    if (cs == null) return null;
    Set<DType> combined = null;
    for (int i = 0; i < cs.getCallSiteRefs().length; i++) {
      CallSiteReference siteRef = cs.getCallSiteRefs()[i];
      IMethod callerMethod = cs.getMethods()[i];
      for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(this.getNode());
          it.hasNext(); ) {
        CGNode caller = it.next();
        if (!caller.getMethod().equals(callerMethod)) continue;
        for (SSAAbstractInvokeInstruction callInstr : caller.getIR().getCalls(siteRef)) {
          if (!(callInstr instanceof PythonInvokeInstruction)) continue;
          PythonInvokeInstruction pyCall = (PythonInvokeInstruction) callInstr;
          int argVn = -1;
          if (paramName != null) argVn = pyCall.getUse(paramName);
          if (argVn == -1 && paramPos >= 0) {
            int numPosParams = pyCall.getNumberOfPositionalParameters() - 1;
            if (paramPos < numPosParams) argVn = pyCall.getUse(paramPos + 1);
          }
          if (argVn <= 0) continue;
          try {
            Set<DType> argDTypes = this.getDTypes(builder, caller, argVn);
            if (argDTypes != null && !argDTypes.isEmpty()) {
              if (combined == null) combined = EnumSet.noneOf(DType.class);
              combined.addAll(argDTypes);
            }
          } catch (IllegalArgumentException e) {
            LOGGER.log(Level.FINE, "Could not get dtypes for caller argument: " + argVn, e);
          }
        }
      }
    }
    return combined;
  }

  /**
   * Retrieves the points-to set for a specific argument of the function call represented by this
   * generator (or specifically for the given node).
   *
   * <p>This method employs a multi-staged strategy to resolve arguments:
   *
   * <ol>
   *   <li><b>Direct Invoke Instruction:</b> If the invoke instruction is directly available (via
   *       {@link #getInvokeInstruction()}), it resolves the argument using the parameter name (for
   *       keyword arguments) or position.
   *   <li><b>Direct Binary Op Instruction:</b> If the instruction is a binary operation (via {@link
   *       #getBinaryOpInstruction()}), it resolves the argument using the position (0 for left, 1
   *       for right).
   *   <li><b>`read_data` Wrapper Handling:</b> If the node corresponds to a `read_data` method
   *       (common in TensorFlow synthetic models), it delegates the resolution to the preceding
   *       `do` method, which is the actual entry point for the operation logic.
   *   <li><b>Context-Sensitive Caller Analysis:</b> It utilizes the {@link CallString} context to
   *       identify the specific caller of the node. It then inspects the call sites in that caller
   *       to find the {@link PythonInvokeInstruction} that targets this node. This is crucial for
   *       distinguishing between different calls to the same operation in a context-sensitive
   *       manner.
   *   <li><b>Callee Parameter Fallback:</b> If the argument cannot be resolved from the caller
   *       (e.g., due to analysis imprecision or manual node creation), it attempts to resolve it
   *       directly from the parameter value numbers within the callee node itself.
   * </ol>
   *
   * @param builder the propagation call graph builder
   * @param node the call graph node representing the function execution
   * @param paramPos the 0-based index of the positional parameter (excluding 'self' for instance
   *     methods)
   * @param paramName the name of the keyword parameter
   * @return the points-to set of the argument, or {@link OrdinalSet#empty()} if not found
   */
  protected OrdinalSet<InstanceKey> getArgumentPointsToSet(
      PropagationCallGraphBuilder builder, CGNode node, int paramPos, String paramName) {
    // Strategy 1: Direct Invoke Instruction
    // If we have direct access to the PythonInvokeInstruction, use it. This is the most direct and
    // reliable method
    // when the generator is instantiated with a source that directly points to the result of an
    // invoke.
    PythonInvokeInstruction call = getInvokeInstruction();
    if (call != null) {
      int argValNum = -1;

      // Try to resolve by name (keyword argument) first.
      if (paramName != null) {
        argValNum = call.getUse(paramName);
      }

      // If not found by name, try by position.
      if (argValNum == -1) {
        if (paramPos == RECEIVER_PARAMETER_POSITION) {
          argValNum = getReceiverValueNumber(node, call);
        } else if (paramPos >= 0) {
          // Adjust position to account for keyword arguments which are stored at the end of the use
          // list.
          int numKeywords = call.getKeywords() != null ? call.getKeywords().size() : 0;
          // Total uses minus the function object itself (index 0) and the keyword args.
          int numPosParams = call.getNumberOfUses() - 1 - numKeywords;

          if (paramPos < numPosParams) {
            // Positional arguments start at index 1 (index 0 is the function object).
            argValNum = call.getUse(paramPos + 1);
          }
        }
      }

      if (argValNum > 0) {
        PointerKey argPk =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, argValNum);
        OrdinalSet<InstanceKey> argPts = builder.getPointerAnalysis().getPointsToSet(argPk);
        if (argPts != null && !argPts.isEmpty()) {
          return argPts;
        }
      }
      return OrdinalSet.empty();
    }

    // Strategy 1.5: Direct Binary Op Instruction
    SSABinaryOpInstruction binOp = getBinaryOpInstruction();
    if (binOp != null) {
      int argValNum = -1;
      if (paramPos >= 0 && paramPos < binOp.getNumberOfUses()) {
        argValNum = binOp.getUse(paramPos);
      }
      if (argValNum > 0) {
        PointerKey argPk =
            builder.getPointerAnalysis().getHeapModel().getPointerKeyForLocal(node, argValNum);
        return builder.getPointerAnalysis().getPointsToSet(argPk);
      }
      return OrdinalSet.empty();
    }

    // Strategy 2: `read_data` Wrapper Handling
    // Synthetic TensorFlow models often use a `read_data` helper method. If we are analyzing such a
    // node,
    // we need to step back to the caller (usually the `do` method of the operation) to find the
    // actual arguments.
    if (node.getMethod().getName().toString().equals(TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME)) {
      OrdinalSet<InstanceKey> ret = OrdinalSet.empty();
      Iterator<CGNode> preds = builder.getCallGraph().getPredNodes(node);
      while (preds.hasNext()) {
        CGNode pred = preds.next();
        if (pred.getMethod().getName().toString().equals(DO_METHOD_NAME)) {
          ret = OrdinalSet.unify(ret, getArgumentPointsToSet(builder, pred, paramPos, paramName));
        }
      }
      return ret;
    }

    // Strategy 3: Context-Sensitive Caller Analysis
    // Use the CallString from the node's context to identify exactly which call site invoked this
    // function.
    // This allows us to look up the arguments passed at that specific call site in the caller's IR.
    CallString cs = (CallString) node.getContext().get(CALL_STRING);
    if (cs != null) {
      OrdinalSet<InstanceKey> combinedPts = OrdinalSet.empty();
      boolean found = false;
      boolean callAnalyzed = false;

      for (int i = 0; i < cs.getCallSiteRefs().length; i++) {
        CallSiteReference siteReference = cs.getCallSiteRefs()[i];
        IMethod callerMethod = cs.getMethods()[i];
        LOGGER.finer(
            "Strategy 3: Checking call site: " + siteReference + " in method: " + callerMethod);

        for (Iterator<CGNode> it = builder.getCallGraph().getPredNodes(node); it.hasNext(); ) {
          CGNode caller = it.next();

          // Ensure we are looking at the caller that matches the method in our call string context.
          if (!caller.getMethod().equals(callerMethod)) {
            continue;
          }

          SSAAbstractInvokeInstruction[] calls = caller.getIR().getCalls(siteReference);
          for (SSAAbstractInvokeInstruction callInstr : calls) {
            // Confirm that this instruction can actually target the current node.
            // This handles polymorphism where a call site might have multiple possible targets.
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

              // Try to resolve by name (keyword argument).
              if (paramName != null) {
                argValNum = pyCallInstr.getUse(paramName);
              }

              // Try to resolve by position.
              if (argValNum == -1) {
                if (paramPos == RECEIVER_PARAMETER_POSITION) {
                  argValNum = getReceiverValueNumber(caller, pyCallInstr);
                } else if (paramPos >= 0) {
                  int numPosParams =
                      pyCallInstr.getNumberOfPositionalParameters() - 1; // Exclude function.
                  if (paramPos < numPosParams) {
                    argValNum =
                        pyCallInstr.getUse(paramPos + 1); // Positional arguments start at index 1.
                  }
                }
              }

              if (argValNum > 0) {
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
      // If we analyzed the call but couldn't find the argument, it likely wasn't provided.
      if (callAnalyzed) {
        return OrdinalSet.empty();
      }
    }

    // Strategy 4: Callee Parameter Fallback
    // If we couldn't resolve the argument from the caller, we look at the parameter value numbers
    // within the callee (the `node` itself). This assumes the argument was successfully passed
    // and mapped to a local variable in the callee.
    int valNum = this.getArgumentValueNumber(node, paramPos);
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
    if (this.getNode()
        .getMethod()
        .getName()
        .toString()
        .equals(TENSOR_GENERATOR_SYNTHETIC_FUNCTION_NAME)) {
      // For read_data nodes, we don't have explicit arguments in the IR.
      // Returning MAX_VALUE acts as a sentinel to bypass the "missing argument" check below
      // and allows getDTypes/getShapes to proceed to getArgumentPointsToSet,
      // which correctly delegates argument resolution to the caller (do).
      return Integer.MAX_VALUE;
    }

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
    }

    SSABinaryOpInstruction binOp = getBinaryOpInstruction();
    if (binOp != null) {
      if (paramPos >= 0 && paramPos < binOp.getNumberOfUses()) {
        return binOp.getUse(paramPos);
      }
      if (optional) return -1;
      throw new IllegalStateException(
          "Cannot determine value number for binary op parameter at position " + paramPos);
    } else {
      // Fallback for manual nodes (no invoke instruction).
      // We assume the arguments are available as parameters in the method body.
      if (paramPos >= 0) {
        return this.getArgumentValueNumber(this.getNode(), paramPos);
      }
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
  protected static TensorGenerator createManualGenerator(
      CGNode node, PropagationCallGraphBuilder builder) {
    TypeReference type = node.getMethod().getDeclaringClass().getReference();
    LOGGER.fine("createManualGenerator checking type: " + type.getName());

    // sanitize the type name by removing the artificial suffix that is added for synthetic
    // classes to facilitate trampoline generation.
    type = sanitize(type);

    LOGGER.fine("createManualGenerator checking sanitized type: " + type.getName());

    if (type.equals(TensorFlowTypes.ONES.getDeclaringClass())) {
      return new Ones(node);
    } else if (type.equals(TensorFlowTypes.SPARSE_EYE.getDeclaringClass())) {
      return new SparseEye(node);
    } else if (type.equals(TensorFlowTypes.EYE.getDeclaringClass())) {
      return new Eye(node);
    } else if (type.equals(TensorFlowTypes.UNIFORM.getDeclaringClass())) {
      return new Uniform(node);
    } else if (type.equals(TensorFlowTypes.NORMAL.getDeclaringClass())) {
      return new Normal(node);
    } else if (type.equals(TensorFlowTypes.TRUNCATED_NORMAL.getDeclaringClass())) {
      return new TruncatedNormal(node);
    } else if (type.equals(TensorFlowTypes.GAMMA.getDeclaringClass())) {
      return new Gamma(node);
    } else if (type.equals(TensorFlowTypes.POISSON.getDeclaringClass())) {
      return new Poisson(node);
    } else if (type.equals(TensorFlowTypes.DATASET_FROM_TENSOR_SLICES_TYPE)
        || type.getName().toString().equals("Ltensorflow/data/from_tensor_slices")) {
      return new DatasetFromTensorSlicesGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_FROM_TENSORS_TYPE)
        || type.getName().toString().equals("Ltensorflow/data/from_tensors")) {
      return new DatasetFromTensorsGenerator(node);
    } else if (type.equals(DATASET_CHOOSE_FROM_DATASETS_TYPE)) {
      return new DatasetChooseFromDatasetsGenerator(node);
    } else if (type.equals(DATASET_SAMPLE_FROM_DATASETS_TYPE)) {
      return new DatasetSampleFromDatasetsGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_FROM_GENERATOR_TYPE)) {
      return new DatasetFromGeneratorGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_ZIP_TYPE)) {
      return new DatasetZipGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_RANGE_TYPE)) {
      return new DatasetRangeGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_RANDOM_TYPE)) {
      return new DatasetRandomGenerator(node);
    } else if (type.equals(TensorFlowTypes.DATASET_BATCH_TYPE)) {
      return new DatasetBatchGenerator(node);
    } else if (type.equals(TensorFlowTypes.IMAGE_DATA_GENERATOR_FLOW_FROM_DIRECTORY_TYPE)) {
      return new FlowFromDirectoryGenerator(node);
    } else if (type.getName().toString().startsWith(TensorFlowTypes.DATA_PACKAGE_PREFIX)) {
      return new DatasetGenerator(node);
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
    } else if (type.equals(TensorFlowTypes.DENSE_CALL.getDeclaringClass())) {
      return new DenseCall(node);
    } else if (type.equals(TensorFlowTypes.MODEL_CALL.getDeclaringClass())) {
      return new ModelCall(node);
    } else if (type.equals(TensorFlowTypes.MODEL.getDeclaringClass())) {
      return new Model(node);
    } else if (type.equals(TensorFlowTypes.INPUT.getDeclaringClass())) {
      return new Input(node);
    }
    return null;
  }
}
