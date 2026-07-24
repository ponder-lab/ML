package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code scipy.sparse} densification ({@code A.todense()}). A SciPy sparse matrix is
 * two-dimensional by construction, so the dense result is rank-2 with extents the analysis cannot
 * compute ({@link UnresolvedDim}, wala/ML#721) and a dtype the untyped sparse operand does not
 * carry. The recovered rank is what lets a consuming allocation (NLPGNN's {@code
 * np.array(features.todense(), dtype=np.float32)}) carry a rank-2 type through to the GNN {@code
 * call} parameters instead of shape-⊤ (wala/ML#768).
 *
 * @see <a
 *     href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.todense.html">scipy.sparse.csr_matrix.todense</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class SparseDensify extends TensorGenerator {

  public SparseDensify(PointsToSetVariable source) {
    super(source);
  }

  public SparseDensify(CGNode node) {
    super(node);
  }

  /**
   * The dense form of a SciPy sparse matrix is always rank 2; both extents are fixed runtime
   * integers the analysis cannot compute ({@link UnresolvedDim}, wala/ML#721).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The singleton rank-2 unresolved shape.
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    return Set.of(List.of(UnresolvedDim.INSTANCE, UnresolvedDim.INSTANCE));
  }

  /**
   * The sparse operand is a SciPy-internal value the analysis never types, so the dense result's
   * dtype is unknown.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The unknown dtype, singleton.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    return EnumSet.of(DType.UNKNOWN);
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
   * Returns the producing library of the modeled value: a densified SciPy matrix is a dense NumPy
   * value (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
