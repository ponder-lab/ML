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
import java.util.Iterator;
import java.util.List;
import java.util.Map;
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
    Compound
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
      return "D:" + type() + "," + value();
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

  private final String cellType;
  private final List<Dimension<?>> dims;

  public TensorType(String cellType, List<Dimension<?>> dims) {
    this.cellType = cellType;
    this.dims = dims;
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

    JsonElement inner = cellType;
    for (Dimension<?> dim : this.getDims()) {
      inner = dim.toJsonSchema(inner);
    }
    return inner;
  }

  public String toMDString() {
    final String dimString =
        getDims().stream().map(Dimension::toMDString).collect(Collectors.joining(" ; "));

    return "[ " + dimString + " **of** _" + getCellType() + "_ ]";
  }

  public String toCString(boolean useMarkdown) {
    final String dimString =
        getDims().stream()
            .map(x -> x.toCString(useMarkdown))
            .map(x -> "[" + x + "]")
            .collect(Collectors.joining());

    final String ctypeString;
    if (useMarkdown) {
      ctypeString = "_" + getCellType() + "_";
    } else {
      ctypeString = getCellType();
    }

    return ctypeString + dimString;
  }

  @Override
  public String toString() {
    return "{" + getDims().toString() + " of " + getCellType() + "}";
  }

  @Override
  public int hashCode() {
    final int prime = 31;
    int result = 1;
    result = prime * result + ((getCellType() == null) ? 0 : getCellType().hashCode());
    result = prime * result + ((getDims() == null) ? 0 : getDims().hashCode());
    return result;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) return true;
    if (obj == null) return false;
    if (getClass() != obj.getClass()) return false;
    TensorType other = (TensorType) obj;
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
    return new TensorType(FLOAT32.name().toLowerCase(), Arrays.asList(batch, vec));
  }

  public static TensorType shapeArg(CGNode node, int literalVn) throws IOException {
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
        if (du.getDef(val) != null && node.getMethod() instanceof AstMethod) {
          Position p =
              ((AstMethod) node.getMethod())
                  .debugInfo()
                  .getInstructionPosition(du.getDef(val).iIndex());
          System.err.println(p);
          SourceBuffer b = new SourceBuffer(p);
          String expr = b.toString();
          System.err.println(expr);
          Integer ival = PythonInterpreter.interpretAsInt(expr);
          if (ival != null) {
            dims.put(index, new NumericDim(ival));
            continue;
          }
        }
        dims.put(index, new SymbolicDim("?"));
      }
    }
    return new TensorType(FLOAT32.name().toLowerCase(), new ArrayList<>(dims.values()));
  }

  @Override
  public Iterator<Dimension<?>> iterator() {
    return getDims().iterator();
  }

  public int symbolicDims() {
    int sz = 0;
    for (Dimension<?> d : this) {
      sz += d.symbolicDims();
    }
    return sz;
  }

  public int concreteSize() {
    int size = -1;
    for (Dimension<?> x : this) {
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
    return cellType;
  }
}
