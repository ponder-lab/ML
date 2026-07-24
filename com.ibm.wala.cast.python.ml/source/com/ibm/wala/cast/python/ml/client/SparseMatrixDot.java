package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.EnumSet;
import java.util.List;
import java.util.Set;

/**
 * Generator for {@code scipy.sparse} matrix products ({@code A.dot(b)}, e.g. on a {@code
 * sp.diags(...)} result). The product of a sparse matrix and a dense array is a dense array whose
 * dtype follows the dense operand {@code b} and whose shape composes the sparse operand's
 * unresolved row extent with {@code b}'s trailing extent; the sparse operand {@code A} is a
 * SciPy-internal value the analysis does not type. See wala/ML#766: the NLPGNN {@code Planetoid}
 * loader's row normalization {@code r_mat_inv.dot(features)} is the real feed of the GNN {@code
 * call} parameters, and without this generator the normalized arm of the feed φ types to ⊥.
 *
 * @see <a
 *     href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.dot.html">scipy.sparse.csr_matrix.dot</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class SparseMatrixDot extends PassThroughUnaryTensorGenerator {

  public SparseMatrixDot(PointsToSetVariable source) {
    super(source);
  }

  public SparseMatrixDot(CGNode node) {
    super(node);
  }

  @Override
  protected int getInputParameterPosition() {
    return 0;
  }

  @Override
  protected String getInputParameterName() {
    return "b";
  }

  /**
   * Derives the product's shape from the dense operand: {@code A.dot(b)} has shape {@code (A.rows,
   * b.cols)} for a rank-2 {@code b} and {@code (A.rows,)} for a rank-1 {@code b}. The sparse
   * operand's row extent is a fixed runtime integer the analysis cannot compute (the sparse operand
   * is a SciPy-internal value it never types), so that axis is {@link UnresolvedDim} (wala/ML#721);
   * the trailing axis carries over from {@code b} when its shape resolves.
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The set of possible output shapes, or {@code null} (⊤) when the dense operand's shape
   *     is unknown or no candidate has rank 1 or 2 (a SciPy sparse product is defined over
   *     rank-1/rank-2 dense operands only).
   */
  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> inputShapes =
        this.shapesOfArg(builder, this.getInputParameterPosition(), this.getInputParameterName());
    if (inputShapes == null) return null;
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (List<Dimension<?>> inputShape : inputShapes) {
      if (inputShape.size() == 1) ret.add(List.of(UnresolvedDim.INSTANCE));
      else if (inputShape.size() == 2) ret.add(List.of(UnresolvedDim.INSTANCE, inputShape.get(1)));
    }
    return ret.isEmpty() ? null : ret;
  }

  /**
   * Collapse-safe record view (wala/ML#718): this generator transforms its input shapes in {@link
   * #getDefaultShapes}, which the base's pass-through identity record path would bypass, so the
   * record view routes through the legacy transform until a member-wise upgrade.
   *
   * @param builder The propagation call graph builder.
   * @return The transformed result, with any partial input collapsed by the legacy view.
   */
  @Override
  protected ShapeResult getDefaultShapeResult(PropagationCallGraphBuilder builder) {
    return ShapeResult.fromLegacy(this.getDefaultShapes(builder));
  }

  /**
   * This generator discards its input's shape, so forwarding operand shapes would overclaim; the
   * feed carries dtype only (wala/ML#682).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The dtype-only feed over the caller-side dense-operand keys, or {@code null} when none
   *     is located.
   */
  @Override
  protected TypeFeed getTypeFeed(PropagationCallGraphBuilder builder) {
    return this.getTypeFeed(builder, TypeFeedKind.DTYPE_ONLY);
  }

  /**
   * Returns the producing library of the modeled value: a SciPy sparse-by-dense product is a dense
   * NumPy value (wala/ML#724).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return {@link TensorOrigin#NUMPY}, singleton.
   */
  @Override
  protected Set<TensorOrigin> getOrigins(PropagationCallGraphBuilder builder) {
    return EnumSet.of(TensorOrigin.NUMPY);
  }
}
