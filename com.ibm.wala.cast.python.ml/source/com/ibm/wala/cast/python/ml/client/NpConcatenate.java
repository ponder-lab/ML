package com.ibm.wala.cast.python.ml.client;

import static com.ibm.wala.cast.python.types.PythonTypes.list;
import static com.ibm.wala.cast.python.types.PythonTypes.tuple;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;

import com.ibm.wala.cast.ipa.callgraph.AstPointerKeyFactory;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;

/**
 * Generator for {@code np.concatenate(arrays, axis=0)}. The shape is {@link Concat}'s: the arrays
 * joined along {@code axis}, which defaults to {@code 0} as {@code tf.concat}'s does. The dtype
 * differs. TensorFlow requires one dtype across the values and {@link Concat} reads the first
 * element's, but NumPy promotes across all of them, and the promoted dtype is often none of the
 * elements' own: {@code float32} beside {@code int64} is {@code float64}. This generator reads
 * every element and applies NumPy's promotion over the engine's numeric dtypes; a mixture it does
 * not cover reads as unknown.
 *
 * @see <a
 *     href="https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html">numpy.concatenate</a>
 */
public class NpConcatenate extends Concat {

  public NpConcatenate(PointsToSetVariable source) {
    super(source);
  }

  public NpConcatenate(CGNode node) {
    super(node);
  }

  @Override
  protected String getValuesParameterName() {
    return "arrays";
  }

  /**
   * Reads every element's dtypes, where {@link Concat} reads the first element's, and promotes them
   * as NumPy does. As in {@link Concat}, a values member that is not a list or tuple contributes
   * nothing, and a list holding an element that is always {@code None} is no tensor, so the dtype
   * is ⊥ when nothing else is read, the twin of the shape read's ⊥ (wala/ML#961, wala/ML#962).
   *
   * @param builder The {@link PropagationCallGraphBuilder} used to build the call graph.
   * @return The promoted dtypes.
   */
  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    OrdinalSet<InstanceKey> valuesPts =
        this.getArgumentPointsToSet(
            builder, this.getValuesParameterIndex(), this.getValuesParameterName());
    if (valuesPts == null || valuesPts.isEmpty()) return EnumSet.of(DType.UNKNOWN);
    Set<DType> ret = EnumSet.noneOf(DType.class);
    boolean deadElement = false;
    for (InstanceKey valIk : valuesPts) {
      AllocationSiteInNode asin = getAllocationSiteInNode(valIk);
      if (asin == null) continue;
      TypeReference ref = asin.concreteType().getReference();
      if (!(ref.equals(list) || ref.equals(tuple))) continue;
      OrdinalSet<InstanceKey> catalog =
          builder
              .getPointerAnalysis()
              .getPointsToSet(
                  ((AstPointerKeyFactory) builder.getPointerKeyFactory())
                      .getPointerKeyForObjectCatalog(asin));
      int count = integerCatalogSize(catalog);
      if (count == 0) {
        ret.add(DType.UNKNOWN);
        continue;
      }
      OrdinalSet<InstanceKey> firstElemPts = this.getElementPts(builder, asin, catalog, 0);
      if (firstElemPts != null && this.hasDeadElement(builder, asin, catalog, firstElemPts)) {
        deadElement = true;
        continue;
      }
      DType promoted = null;
      for (int i = 0; i < count; i++) {
        OrdinalSet<InstanceKey> elementPts = this.getElementPts(builder, asin, catalog, i);
        Set<DType> dtypes = elementPts == null ? null : this.getDTypesOfValue(builder, elementPts);
        // An element whose dtype is not one known dtype makes the promotion unknown.
        DType element =
            dtypes == null || dtypes.size() != 1 ? DType.UNKNOWN : dtypes.iterator().next();
        promoted = promoted == null ? element : promote(promoted, element);
      }
      ret.add(promoted);
    }
    if (ret.isEmpty() && deadElement) return Collections.emptySet();
    return ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret;
  }

  /**
   * NumPy's promotion of two array dtypes over the engine's numeric dtypes: {@code bool} gives way
   * to anything; integers take the wider, and {@code uint8} gives way to any signed integer; {@code
   * float64} beside a real dtype is {@code float64}; {@code float32} beside {@code int32} or {@code
   * int64} is {@code float64}, and beside {@code uint8} or {@code bool} stays {@code float32}; a
   * complex dtype beside a real one is {@code complex64} only when the real one fits it ({@code
   * bool}, {@code uint8} or {@code float32}) and {@code complex128} otherwise. Any other pair, one
   * that is non-numeric or unknown, is unknown.
   *
   * @param a One dtype.
   * @param b The other dtype.
   * @return The promoted dtype.
   */
  static DType promote(DType a, DType b) {
    if (a == b) return a;
    if (a == DType.UNKNOWN || b == DType.UNKNOWN) return DType.UNKNOWN;
    if (a == DType.BOOL && b.isNumeric()) return b;
    if (b == DType.BOOL && a.isNumeric()) return a;
    if (!a.isNumeric() || !b.isNumeric()) return DType.UNKNOWN;
    if (isComplex(a) || isComplex(b)) {
      if (a == DType.COMPLEX128 || b == DType.COMPLEX128) return DType.COMPLEX128;
      DType real = isComplex(a) ? b : a;
      return real == DType.UINT8 || real == DType.FLOAT32 ? DType.COMPLEX64 : DType.COMPLEX128;
    }
    if (a == DType.FLOAT64 || b == DType.FLOAT64) return DType.FLOAT64;
    if (a == DType.FLOAT32 || b == DType.FLOAT32) {
      DType other = a == DType.FLOAT32 ? b : a;
      return other == DType.UINT8 ? DType.FLOAT32 : DType.FLOAT64;
    }
    // Both integral.
    if (a == DType.INT64 || b == DType.INT64) return DType.INT64;
    if (a == DType.INT32 || b == DType.INT32) return DType.INT32;
    return DType.UNKNOWN;
  }

  private static boolean isComplex(DType dtype) {
    return dtype == DType.COMPLEX64 || dtype == DType.COMPLEX128;
  }
}
