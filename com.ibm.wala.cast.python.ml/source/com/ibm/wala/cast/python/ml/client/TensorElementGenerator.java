package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonPropertyRead;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for elements of a tensor, used during iteration.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class TensorElementGenerator extends TensorGenerator implements DelegatingTensorGenerator {

  private final TensorGenerator containerGenerator;

  /**
   * Separates elements of different containers read by one subscript, or {@code null} when the
   * subscript reads one container.
   */
  private final Object containerDiscriminator;

  public TensorElementGenerator(PointsToSetVariable source, TensorGenerator containerGenerator) {
    this(source, containerGenerator, false);
  }

  /**
   * Constructs an element of the given container, optionally told apart from the elements of the
   * other containers one subscript reads. A subscript whose receiver has several creators reads an
   * element of each, one generator per creator anchored on the same subscript, and the memo key of
   * an evaluation is the class and the anchor, so without the discriminator every part would replay
   * the first part's result.
   *
   * @param source The subscript's {@link PointsToSetVariable}.
   * @param containerGenerator The container's generator.
   * @param perContainer Whether this element is one of several the subscript reads.
   */
  TensorElementGenerator(
      PointsToSetVariable source, TensorGenerator containerGenerator, boolean perContainer) {
    super(source);
    this.containerGenerator = containerGenerator;
    this.containerDiscriminator =
        perContainer && containerGenerator != null ? evaluationKey(containerGenerator) : null;
  }

  @Override
  protected Object operationDiscriminator() {
    return this.containerDiscriminator;
  }

  public TensorGenerator getContainerGenerator() {
    return containerGenerator;
  }

  @Override
  public TensorGenerator getUnderlying() {
    return containerGenerator;
  }

  @Override
  public String toString() {
    return "TensorElementGenerator("
        + containerGenerator
        + (containerGenerator == null
            ? ""
            : " [" + containerGenerator.getClass().getSimpleName() + "]")
        + ")";
  }

  /**
   * Returns the shapes of the elements contained within the container tensor.
   *
   * <p>When iterating over a tensor of rank {@code N}, the elements produced have rank {@code N-1},
   * where the first dimension of the container is removed. For example, iterating over a tensor of
   * shape {@code [5, 10]} yields elements of shape {@code [10]}.
   *
   * @param builder The {@link PropagationCallGraphBuilder} for the analysis.
   * @return A set of possible element shapes.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    // The container generator may be null if the factory's recursive walk could not resolve the
    // iterable's source (see wala/ML#363). Return empty set (⊥ / not-a-tensor) so the source is
    // effectively dropped from tensor analysis, matching the pre-audit behaviour where the IAE
    // bubbled up through the wrapper constructor and the source never got added to the init map.
    if (this.containerGenerator == null) return Collections.emptySet();
    // Calling a functional model built with a list or tuple of outputs returns a list. Its element
    // is the output at that position, not an axis-0 slice of one tensor (wala/ML#1157).
    if (this.containerGenerator instanceof ModelCall call) {
      List<ModelCall.OutputDisplay> positions = call.getSequenceOutputGenerators(builder);
      if (positions != null) return call.getOutputElementShapes(builder, positions, this.index());
    }
    Set<List<Dimension<?>>> containerShapes = this.containerGenerator.getShapes(builder);
    if (containerShapes == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> shape : containerShapes) {
      if (shape.size() > 0) {
        // Peel off the first dimension.
        ret.add(new ArrayList<>(shape.subList(1, shape.size())));
      } else {
        // Iterating a scalar tensor is technically an error in TensorFlow, but we treat its
        // elements
        // as scalars for robustness.
        ret.add(Collections.emptyList());
      }
    }
    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // See the null check in getDefaultShapes. ⊥ preserves source-dropped semantics.
    if (this.containerGenerator == null) return Collections.emptySet();
    if (this.containerGenerator instanceof ModelCall call) {
      List<ModelCall.OutputDisplay> positions = call.getSequenceOutputGenerators(builder);
      if (positions != null) return call.getOutputElementDTypes(builder, positions, this.index());
    }
    return this.containerGenerator.getDTypes(builder);
  }

  /**
   * The constant index this element is read at, as {@code res[1]} or {@code res[-1]} reads it, or
   * {@code null} when it is read by iteration or at an index that is not a constant.
   *
   * @return The constant index, or {@code null}.
   */
  private Integer index() {
    if (this.getSource() == null
        || !(this.getSource().getPointerKey() instanceof LocalPointerKey local)) return null;
    CGNode node = local.getNode();
    if (node.getIR() == null || node.getDU() == null) return null;
    SSAInstruction def = node.getDU().getDef(local.getValueNumber());
    if (!(def instanceof PythonPropertyRead read)) return null;
    SymbolTable symbolTable = node.getIR().getSymbolTable();
    int member = read.getMemberRef();
    if (member <= 0 || !symbolTable.isConstant(member)) return null;
    Object value = symbolTable.getConstantValue(member);
    return value instanceof Integer || value instanceof Long ? ((Number) value).intValue() : null;
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
}
