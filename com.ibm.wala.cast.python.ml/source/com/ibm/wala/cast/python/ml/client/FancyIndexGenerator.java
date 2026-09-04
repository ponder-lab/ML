package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

/**
 * A generator for the result of NumPy fancy indexing: a subscript whose member is itself an array,
 * {@code a[idx]} with {@code idx} an ndarray (wala/ML#866). Unlike an element read, fancy indexing
 * does not peel the receiver's leading axis: it REPLACES it with the index's shape, so {@code
 * np.eye(3)[np.array([0, 1, 2])]} is {@code (3, 3)}, not {@code (3,)}. The scalar spelling {@code
 * a[0]} produces the same output as the old peeling for the same receiver and is correct there,
 * which is why the two spellings must dispatch differently: a fix that merely stopped peeling would
 * trade this defect for that regression.
 *
 * <p>Shapes compose per (index shape, receiver shape) pair as {@code indexShape ++
 * receiverShape[1:]} for integer indexes. A boolean index is a MASK, whose selected count is a
 * runtime fact: a known rank-1 mask keeps the receiver's rank with an {@link UnresolvedDim} leading
 * axis (a fixed runtime integer the analysis cannot compute, per the wala/ML#721 criterion); any
 * other mask form degrades to ⊤, as does an index whose shape is unknown, since an unknown index
 * rank makes the result rank unknown. The dtype is the receiver's.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class FancyIndexGenerator extends TensorGenerator implements DelegatingTensorGenerator {

  private final TensorGenerator containerGenerator;

  private final TensorGenerator indexGenerator;

  public FancyIndexGenerator(
      PointsToSetVariable source,
      TensorGenerator containerGenerator,
      TensorGenerator indexGenerator) {
    super(source);
    this.containerGenerator = containerGenerator;
    this.indexGenerator = indexGenerator;
  }

  @Override
  public TensorGenerator getUnderlying() {
    return containerGenerator;
  }

  @Override
  public String toString() {
    return "FancyIndexGenerator(" + containerGenerator + " indexed by " + indexGenerator + ")";
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> containerShapes = this.containerGenerator.getShapes(builder);
    if (containerShapes == null) return null;

    Set<List<Dimension<?>>> indexShapes = this.indexGenerator.getShapes(builder);
    Set<DType> indexDTypes = this.indexGenerator.getDTypes(builder);
    boolean mask = indexDTypes != null && indexDTypes.contains(DType.BOOL);

    Set<List<Dimension<?>>> ret = HashSetFactory.make();

    for (List<Dimension<?>> containerShape : containerShapes) {
      List<Dimension<?>> tail =
          containerShape.isEmpty()
              ? Collections.emptyList()
              : containerShape.subList(1, containerShape.size());

      if (mask) {
        // Only a known rank-1 mask has an expressible result: the receiver's rank with the
        // selected count unresolved. Any other mask form (higher rank flattens several axes;
        // unknown rank could be either) makes the result rank unknown.
        if (indexShapes == null || indexShapes.stream().anyMatch(s -> s.size() != 1)) return null;

        List<Dimension<?>> shape = new ArrayList<>();
        shape.add(UnresolvedDim.INSTANCE);
        shape.addAll(tail);
        ret.add(shape);
        continue;
      }

      // An unknown index shape means an unknown result rank.
      if (indexShapes == null) return null;

      for (List<Dimension<?>> indexShape : indexShapes) {
        List<Dimension<?>> shape = new ArrayList<>(indexShape);
        shape.addAll(tail);
        ret.add(shape);
      }
    }

    return ret;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return this.containerGenerator.getDTypes(builder);
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
