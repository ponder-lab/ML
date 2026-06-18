/*
 * Copyright (c) 2018 IBM Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 */
package com.ibm.wala.cast.python.ml.types;

import static com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType.FLOAT32;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.cast.python.util.PythonInterpreter;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.cast.util.SourceBuffer;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.ssa.SymbolTable;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;
import java.util.logging.Logger;
import java.util.stream.Collectors;

public class TensorType implements Iterable<Dimension<?>> {

  private static final Logger logger = Logger.getLogger(TensorType.class.getName());

  public enum Format {
    CString,
    MCString,
    MDString,
    JsonSchema
  };

  enum DimensionType {
    Constant,
    Symbolic,
    Compound,
    Ragged,
    Dynamic
  };

  public abstract static class Dimension<T> {
    private final T v;

    protected Dimension(T v) {
      this.v = v;
    }

    abstract DimensionType type();

    abstract int symbolicDims();

    abstract int concreteSize();

    abstract String toMDString();

    abstract String toCString(boolean useMarkdown);

    JsonElement toJsonSchema(JsonElement inner) {
      JsonObject obj = new JsonObject();
      obj.addProperty("type", "array");
      if (inner != null) {
        obj.add("items", inner);
      }
      final int size = concreteSize();

      if (size >= 0) {
        obj.addProperty("minItems", size);
        obj.addProperty("maxItems", size);
        obj.addProperty("description", "Array of dimension " + this.toCString(false));
      }
      return obj;
    }

    public T value() {
      return v;
    }

    @Override
    public String toString() {
      // Skip the ",value" suffix for `Void`-payload sentinels (e.g. `DynamicDim`,
      // `RaggedDim`) whose `value()` is always `null`; the trailing ",null" otherwise
      // collides visually with the ", " dim separator in shape renderings. See
      // wala/ML#558.
      T v = value();
      return v == null ? "D:" + type() : "D:" + type() + "," + v;
    }

    @Override
    public int hashCode() {
      final int prime = 31;
      int result = 1;
      result = prime * result + ((v == null) ? 0 : v.hashCode());
      return result;
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) return true;
      if (obj == null) return false;
      if (getClass() != obj.getClass()) return false;
      Dimension<?> other = (Dimension<?>) obj;
      if (v == null) {
        if (other.v != null) return false;
      } else if (!v.equals(other.v)) return false;
      return true;
    }

    public static Dimension<?> max(Dimension<?> d1, Dimension<?> d2) {
      if (d1 instanceof NumericDim && d2 instanceof NumericDim) {
        Integer v1 = ((NumericDim) d1).value();
        Integer v2 = ((NumericDim) d2).value();

        return new NumericDim(Math.max(v1, v2));
      } else
        throw new IllegalArgumentException(
            "Cannot compute max of non-numeric dimensions: " + d1 + ", " + d2);
    }
  }

  public static class SymbolicDim extends Dimension<String> {
    public SymbolicDim(String name) {
      super(name);
    }

    @Override
    DimensionType type() {
      return DimensionType.Symbolic;
    }

    @Override
    int concreteSize() {
      return -1;
    }

    @Override
    int symbolicDims() {
      return 1;
    }

    @Override
    String toMDString() {
      return "*" + value() + "*";
    }

    @Override
    String toCString(boolean useMarkdown) {
      if (useMarkdown) {
        return "*" + value() + "*";
      } else {
        return value();
      }
    }
  }

  public static class NumericDim extends Dimension<Integer> {
    public NumericDim(Integer v) {
      super(v);
    }

    @Override
    DimensionType type() {
      return DimensionType.Constant;
    }

    @Override
    int concreteSize() {
      return value();
    }

    @Override
    int symbolicDims() {
      return 0;
    }

    @Override
    String toMDString() {
      return value().toString();
    }

    @Override
    String toCString(boolean useMarkdown) {
      return value().toString();
    }
  }

  /**
   * Marker for a ragged dimension &mdash; a dimension whose size varies across rows of a single
   * ragged tensor instance (e.g., the second axis of {@code tf.ragged.constant([[1, 2], [3]])}).
   *
   * <p>Ragged dimensions used to be encoded as raw {@code null} entries in {@code TensorType.dims};
   * the typed sentinel restores the implicit non-null-element contract of {@link
   * TensorType#iterator()} and lets downstream consumers (e.g., Hybridize's input-signature
   * inference) discriminate ragged from "unknown size" (which is still encoded as raw {@code null}
   * for dynamic-batch / placeholder dims, distinct semantics). See <a
   * href="https://github.com/wala/ML/issues/544">wala/ML#544</a>.
   *
   * <p>The {@link Dimension#value() value} is always {@code null} &mdash; raggedness carries no
   * payload beyond its identity. Because every instance is structurally equal to every other, use
   * the shared {@link #INSTANCE} rather than allocating fresh objects.
   */
  public static class RaggedDim extends Dimension<Void> {
    /** Shared singleton; {@code RaggedDim} carries no per-instance state. */
    public static final RaggedDim INSTANCE = new RaggedDim();

    /**
     * @implNote Private — all callers should use {@link #INSTANCE}. The {@code super(null)} call is
     *     mechanical: {@link Dimension} requires a value of type {@code T}, and {@code Void} has no
     *     instances, so {@code null} is the only legal argument. Raggedness carries no payload
     *     beyond the type's identity.
     */
    private RaggedDim() {
      super(null);
    }

    @Override
    DimensionType type() {
      return DimensionType.Ragged;
    }

    @Override
    int concreteSize() {
      return -1;
    }

    @Override
    int symbolicDims() {
      return 1;
    }

    @Override
    String toMDString() {
      return "*ragged*";
    }

    @Override
    String toCString(boolean useMarkdown) {
      if (useMarkdown) {
        return "*ragged*";
      } else {
        return "ragged";
      }
    }

    /**
     * Override {@link Dimension#hashCode} (which hashes only the payload {@code value}) so that
     * null-payload singletons of different concrete classes don't collide. {@link Dimension#equals}
     * already distinguishes by {@code getClass()}, so the bucket-level separation here is a
     * hash-performance fix, not a correctness one.
     */
    @Override
    public int hashCode() {
      return RaggedDim.class.hashCode();
    }

    /**
     * Override paired with {@link #hashCode} to satisfy the {@code equals}/{@code hashCode}
     * contract under CodeQL's {@code java/inconsistent-equals-and-hashcode}. {@code RaggedDim} is a
     * singleton, so equality reduces to instance identity — any reachable {@code RaggedDim}
     * reference is {@link #INSTANCE}.
     */
    @Override
    public boolean equals(Object obj) {
      return this == obj;
    }
  }

  /**
   * Marker for a dynamic dimension&mdash;a dimension whose size is uniform within a single tensor
   * instance but unknown statically (e.g., the batch axis of {@code tf.keras.Input(shape=(None,
   * 4))}, or any axis explicitly modeled as {@code None} in a Keras input signature).
   *
   * <p>Dynamic dimensions used to be encoded as raw {@code null} entries in {@code
   * TensorType.dims}; the typed sentinel restores the implicit non-null-element contract of {@link
   * TensorType#iterator()} and discriminates "dynamic/batch/placeholder" from ragged (where the
   * size varies across rows of a single tensor instance, see {@link RaggedDim}). See <a
   * href="https://github.com/wala/ML/issues/545">wala/ML#545</a>.
   *
   * <p>The {@link Dimension#value() value} is always {@code null}&mdash;dynamic-ness carries no
   * payload beyond its identity. Because every instance is structurally equal to every other, use
   * the shared {@link #INSTANCE} rather than allocating fresh objects.
   */
  public static class DynamicDim extends Dimension<Void> {
    /** Shared singleton; {@code DynamicDim} carries no per-instance state. */
    public static final DynamicDim INSTANCE = new DynamicDim();

    /**
     * @implNote Private&mdash;all callers should use {@link #INSTANCE}. The {@code super(null)}
     *     call is mechanical: {@link Dimension} requires a value of type {@code T}, and {@code
     *     Void} has no instances, so {@code null} is the only legal argument. Dynamic-ness carries
     *     no payload beyond the type's identity.
     */
    private DynamicDim() {
      super(null);
    }

    @Override
    DimensionType type() {
      return DimensionType.Dynamic;
    }

    @Override
    int concreteSize() {
      return -1;
    }

    @Override
    int symbolicDims() {
      return 1;
    }

    @Override
    String toMDString() {
      return "*dynamic*";
    }

    @Override
    String toCString(boolean useMarkdown) {
      if (useMarkdown) {
        return "*dynamic*";
      } else {
        return "dynamic";
      }
    }

    /**
     * Override {@link Dimension#hashCode} (which hashes only the payload {@code value}) so that
     * null-payload singletons of different concrete classes don't collide. See {@link
     * RaggedDim#hashCode} for the shared rationale.
     */
    @Override
    public int hashCode() {
      return DynamicDim.class.hashCode();
    }

    /**
     * Override paired with {@link #hashCode} to satisfy the {@code equals}/{@code hashCode}
     * contract under CodeQL's {@code java/inconsistent-equals-and-hashcode}. See {@link
     * RaggedDim#equals} for the shared singleton-identity rationale.
     */
    @Override
    public boolean equals(Object obj) {
      return this == obj;
    }
  }

  public static class CompoundDim extends Dimension<List<Dimension<?>>> {
    public CompoundDim(List<Dimension<?>> v) {
      super(v);
    }

    @Override
    DimensionType type() {
      return DimensionType.Compound;
    }

    @Override
    int concreteSize() {
      int size = -1;
      for (Dimension<?> x : value()) {
        final int xs = x.concreteSize();
        if (xs >= 0) {
          if (size >= 0) {
            size *= xs;
          } else {
            size = xs;
          }
        }
      }
      return size;
    }

    @Override
    int symbolicDims() {
      int size = 0;
      for (Dimension<?> x : value()) {
        size += x.symbolicDims();
      }
      return size;
    }

    @Override
    String toMDString() {
      return value().stream().map(Dimension::toMDString).collect(Collectors.joining(" \\* "));
    }

    @Override
    String toCString(boolean useMarkdown) {
      final String delim;
      if (useMarkdown) {
        delim = " \\* ";
      } else {
        delim = " * ";
      }

      return value().stream().map(x -> x.toCString(useMarkdown)).collect(Collectors.joining(delim));
    }
  }

  /**
   * The dtype as a typed enum value — every {@code TensorType} has one. The {@link
   * #TensorType(DType, List)} ctor stores the passed value directly; the {@link #TensorType(String,
   * List)} ctor parses the cellType string at construction and throws {@link
   * IllegalArgumentException} if it doesn't map to a known {@link DType}. See <a
   * href="https://github.com/wala/ML/issues/533">wala/ML#533</a>.
   */
  private final DType dtype;

  private final List<Dimension<?>> dims;

  /**
   * The storage layout of a tensor. Sparseness is a tensor-level storage property orthogonal to
   * {@link #dims} (a sparse tensor has the same dense shape as its dense counterpart), so it is
   * modeled as the concrete type ({@link SparseTensorType}) rather than a {@link Dimension}
   * (contrast {@link RaggedDim}, which is genuinely per-axis). A consumer reading a per-parameter
   * {@code TensorType} can branch on {@link #layout()} to emit the right spec. See <a
   * href="https://github.com/wala/ML/issues/588">wala/ML#588</a>.
   */
  public enum Layout {
    DENSE,
    SPARSE
  }

  /**
   * Constructs a {@code TensorType} from a {@code cellType} string. The string must map to a known
   * {@link DType} via {@code DType.valueOf(cellType.toUpperCase(Locale.ROOT))}; an unparseable
   * value is rejected at construction time. Prefer {@link #TensorType(DType, List)} when a typed
   * {@link DType} is already in hand. The resulting type is dense; derive a sparse one with {@link
   * #asSparse()}.
   *
   * @param cellType The lowercase dtype name (e.g., {@code "float32"}). Must not be null and must
   *     parse to a {@link DType} constant.
   * @param dims The dimensions of the tensor; may be null to indicate unknown rank (⊤ shape).
   * @throws IllegalArgumentException if {@code cellType} doesn't map to a {@link DType}.
   */
  public TensorType(String cellType, List<Dimension<?>> dims) {
    // A TensorType with a null cellType is nonsensical: every tensor has a dtype, even if it is
    // DType.UNKNOWN. Dims, on the other hand, may legitimately be null (unknown rank / ⊤ shape).
    try {
      this.dtype =
          DType.valueOf(
              Objects.requireNonNull(cellType, "Cell type must not be null.")
                  .toUpperCase(Locale.ROOT));
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException(
          "Cell type: " + cellType + " does not map to a known " + DType.class.getName() + ".", e);
    }
    this.dims = dims;
  }

  /**
   * Constructs a dense {@code TensorType}. The dtype is the internal source of truth since
   * wala/ML#533; the cellType String exposed via {@link #getCellType} is derived from it on demand.
   *
   * @param dtype The tensor element type. Must not be null.
   * @param dims The dimensions of the tensor; may be null to indicate unknown rank (⊤ shape).
   */
  public TensorType(DType dtype, List<Dimension<?>> dims) {
    this.dtype = Objects.requireNonNull(dtype, "TensorType dtype must not be null");
    this.dims = dims;
  }

  /**
   * The storage layout of this tensor: the polymorphic source of truth for sparseness. Overridden
   * by {@link SparseTensorType}; the base {@code TensorType} is always {@link Layout#DENSE}.
   *
   * @return {@link Layout#DENSE} for a dense tensor, {@link Layout#SPARSE} for a sparse one.
   */
  public Layout layout() {
    return Layout.DENSE;
  }

  /**
   * Convenience predicate derived from {@link #layout()}.
   *
   * @return {@code true} if this is a sparse tensor ({@link Layout#SPARSE}), {@code false} if
   *     dense.
   */
  public final boolean isSparse() {
    return this.layout() == Layout.SPARSE;
  }

  /**
   * Returns a sparse view of this type: the same dtype and dims, as a {@link SparseTensorType}.
   * Returns {@code this} when already sparse.
   *
   * @return A {@link SparseTensorType} with this type's dtype and dims, or {@code this} if already
   *     sparse.
   */
  public TensorType asSparse() {
    return this.isSparse() ? this : new SparseTensorType(this.dtype, this.dims);
  }

  String toFormattedString(Format fmt) {
    switch (fmt) {
      case CString:
        return toCString(false);
      case MCString:
        return toCString(true);
      case MDString:
        return toMDString();
      case JsonSchema:
        return new Gson().toJson(toJsonSchema());
      default:
        throw new IllegalArgumentException("unknown format type: " + fmt);
    }
  }

  public JsonElement toJsonSchema() {
    JsonObject cellType = null;
    if (this.getCellType() != null) {
      cellType = new JsonObject();
      cellType.addProperty("description", "Elements of type " + this.getCellType());
    }

    if (this.getDims() == null) {
      JsonObject unknownShape = new JsonObject();
      unknownShape.addProperty(
          "description", "Unknown shape of elements of type " + this.getCellType());
      return unknownShape;
    }

    JsonElement inner = cellType;
    for (Dimension<?> dim : this.getDims()) {
      inner = dim.toJsonSchema(inner);
    }
    return inner;
  }

  public String toMDString() {
    final String dimString;
    if (getDims() == null) {
      dimString = "?";
    } else {
      dimString = getDims().stream().map(Dimension::toMDString).collect(Collectors.joining(" ; "));
    }

    return "[ "
        + dimString
        + " **of** _"
        + getCellType()
        + "_ ]"
        + (this.isSparse() ? " (sparse)" : "");
  }

  public String toCString(boolean useMarkdown) {
    final String dimString;
    if (getDims() == null) {
      dimString = "[?]";
    } else {
      dimString =
          getDims().stream()
              .map(x -> x.toCString(useMarkdown))
              .map(x -> "[" + x + "]")
              .collect(Collectors.joining());
    }

    final String ctypeString;
    if (useMarkdown) {
      ctypeString = "_" + getCellType() + "_";
    } else {
      ctypeString = getCellType();
    }

    return ctypeString + dimString + (this.isSparse() ? " (sparse)" : "");
  }

  @Override
  public String toString() {
    return "{"
        + (getDims() == null ? "?" : getDims().toString())
        + " of "
        + getCellType()
        + (this.isSparse() ? " (sparse)" : "")
        + "}";
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((getCellType() == null) ? 0 : getCellType().hashCode());
    result = prime * result + ((getDims() == null) ? 0 : getDims().hashCode());
    result = prime * result + (this.isSparse() ? 1231 : 1237);
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (getClass() != obj.getClass()) return false;
    TensorType other = (TensorType) obj;
    // The getClass() check above already separates a dense TensorType from a SparseTensorType, so
    // their layouts need not be compared again here.
    if (getCellType() == null) {
      if (other.getCellType() != null) return false;
    } else if (!getCellType().equals(other.getCellType())) return false;
    if (getDims() == null) {
      if (other.getDims() != null) return false;
    } else if (!getDims().equals(other.getDims())) return false;
    return true;
  }

  public static TensorType mnistInput() {
    Dimension<String> batch = new SymbolicDim("n");
    Dimension<Integer> x = new NumericDim(28);
    Dimension<Integer> y = new NumericDim(28);
    Dimension<List<Dimension<?>>> vec = new CompoundDim(Arrays.asList(x, y));
    return new TensorType(FLOAT32.name().toLowerCase(Locale.ROOT), Arrays.asList(batch, vec));
  }

  public static TensorType shapeArg(CGNode node, int literalVn) {
    logger.fine(() -> node.getIR().toString());
    Map<Integer, Dimension<?>> dims = new TreeMap<>();
    DefUse du = node.getDU();
    SymbolTable S = node.getIR().getSymbolTable();
    for (Iterator<SSAInstruction> uses = du.getUses(literalVn); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      int val, ref;
      Integer index = null;

      if (use instanceof SSAPutInstruction) {
        SSAPutInstruction put = (SSAPutInstruction) use;
        if (put.isStatic()) continue;
        val = put.getVal();
        ref = put.getRef();
        try {
          index = Integer.parseInt(put.getDeclaredField().getName().toString());
        } catch (NumberFormatException e) {
          // ignore
        }
      } else if (use instanceof PythonPropertyWrite) {
        val = ((PythonPropertyWrite) use).getValue();
        ref = ((PythonPropertyWrite) use).getObjectRef();
        int indexVn = ((PythonPropertyWrite) use).getMemberRef();
        if (S.isNumberConstant(indexVn)) {
          index = ((Number) S.getConstantValue(indexVn)).intValue();
        } else if (S.isStringConstant(indexVn)) {
          try {
            index = Integer.parseInt(S.getStringValue(indexVn));
          } catch (NumberFormatException e) {
            // ignore
          }
        }
      } else {
        continue;
      }

      if (ref != literalVn) {
        continue;
      }

      if (index == null) {
        // If we can't determine the index, we can't reliably build the shape.
        // But maybe we should just skip this write?
        // Previous behavior just added it.
        // Let's log and skip.
        logger.warning("Could not determine index for shape arg write: " + use);
        continue;
      }

      if (S.isNumberConstant(val)) {
        int v = ((Number) S.getConstantValue(val)).intValue();
        logger.fine("value: " + v);
        dims.put(index, v >= 0 ? new NumericDim((Integer) v) : new SymbolicDim("?"));
      } else {
        // Guard `iIndex() >= 0`: synthetic defs have a negative instruction index,
        // for which `getInstructionPosition` has no valid position (same guard as
        // `PythonTurtleAnalysisEngine`).
        if (du.getDef(val) != null
            && node.getMethod() instanceof AstMethod
            && du.getDef(val).iIndex() >= 0) {
          Position p =
              ((AstMethod) node.getMethod())
                  .debugInfo()
                  .getInstructionPosition(du.getDef(val).iIndex());
          // `SourceBuffer(Position)` reads the underlying source file. If the
          // position is absent (detached def) or the file is unavailable
          // (synthetic position), fall through to the symbolic-dim fallback below
          // rather than crashing the analysis. The `p != null` guard prevents a
          // `SourceBuffer(null)` NPE, which the `IOException` catch would not
          // handle.
          if (p != null) {
            try {
              SourceBuffer b = new SourceBuffer(p);
              String expr = b.toString();
              Integer ival = PythonInterpreter.interpretAsInt(expr);
              if (ival != null) {
                dims.put(index, new NumericDim(ival));
                continue;
              }
            } catch (IOException e) {
              logger.fine(() -> "Could not read source for shape-arg position " + p + ": " + e);
            }
          }
        }
        // TODO(https://github.com/wala/ML/issues/581): when the dim is a binary op over
        // constant-valued operands (field reads such as `self.heads * self.out_features`, globals),
        // fold it via the analysis instead of degrading to a symbolic dim. `interpretAsInt` above
        // only handles pure-literal source text, so the generator-side path should own this.
        dims.put(index, new SymbolicDim("?"));
      }
    }
    return new TensorType(FLOAT32.name().toLowerCase(Locale.ROOT), new ArrayList<>(dims.values()));
  }

  @Override
  public Iterator<Dimension<?>> iterator() {
    return getDims() == null ? Collections.emptyIterator() : getDims().iterator();
  }

  public int symbolicDims() {
    int sz = 0;
    for (Dimension<?> d : this) {
      if (d != null) {
        sz += d.symbolicDims();
      }
    }
    return sz;
  }

  public int concreteSize() {
    int size = -1;
    for (Dimension<?> x : this) {
      if (x != null) {
        final int xs = x.concreteSize();
        if (xs >= 0) {
          if (size >= 0) {
            size *= xs;
          } else {
            size = xs;
          }
        }
      }
    }
    return size;
  }

  /**
   * Returns the dimensions of the tensor.
   *
   * @return The dimensions of the tensor.
   */
  public List<Dimension<?>> getDims() {
    return dims;
  }

  /**
   * Returns the cell type of the tensor.
   *
   * @return The cell type of the tensor.
   */
  public String getCellType() {
    return dtype.name().toLowerCase(Locale.ROOT);
  }

  /**
   * Returns the dtype of the tensor as a typed {@link DType} enum value.
   *
   * <p>{@code dtype} is the internal source of truth since wala/ML#533; consumers branch on it as a
   * typed value instead of doing cast-and-uppercase boilerplate on {@link #getCellType()}. The
   * String ctor parses-or-throws at construction, so this accessor never throws.
   *
   * @return The dtype enum value.
   */
  public DType getDType() {
    return dtype;
  }
}
