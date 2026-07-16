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
package com.ibm.wala.cast.python.ml.analysis;

import com.ibm.wala.cast.loader.AstMethod;
import com.ibm.wala.cast.lsp.AnalysisError;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.cast.python.ml.types.TensorType.CompoundDim;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.SymbolicDim;
import com.ibm.wala.cast.tree.CAstSourcePositionMap.Position;
import com.ibm.wala.cast.util.SourceBuffer;
import com.ibm.wala.dataflow.graph.AbstractMeetOperator;
import com.ibm.wala.dataflow.graph.DataflowSolver;
import com.ibm.wala.dataflow.graph.IKilldallFramework;
import com.ibm.wala.dataflow.graph.ITransferFunctionProvider;
import com.ibm.wala.fixpoint.UnaryOperator;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ssa.DefUse;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.graph.Graph;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.eclipse.lsp4j.DiagnosticSeverity;

public class TensorTypeAnalysis extends DataflowSolver<PointsToSetVariable, TensorVariable>
    implements Iterable<Pair<PointerKey, TensorVariable>> {

  static class ReshapeError implements AnalysisError {
    ReshapeError(TensorType from, TensorType to, Position pos, Position definer) {
      this.definer = definer;
      this.from = from;
      this.to = to;
      this.pos = pos;
    }

    TensorType from, to;
    Position pos, definer;

    public Iterable<Pair<Position, String>> related() {
      return Collections.singleton(Pair.make(definer, "definition"));
    }

    @Override
    public Position position() {
      return pos;
    }

    @Override
    public String toString() {
      return toString(false);
    }

    @Override
    public String toString(boolean useMarkdown) {
      return "Cannot reshape " + from.toCString(useMarkdown) + " to " + to.toCString(useMarkdown);
    }

    @Override
    public DiagnosticSeverity severity() {
      return DiagnosticSeverity.Warning;
    }

    @Override
    public String source() {
      return "Ariadne";
    }
  }

  static class ConvError implements AnalysisError {
    ConvError(TensorType from, int dims, Position pos, Position definer) {
      this.definer = definer;
      this.from = from;
      this.dims = dims;
      this.pos = pos;
    }

    Position definer;
    TensorType from;
    int dims;
    Position pos;

    public Iterable<Pair<Position, String>> related() {
      return Collections.singleton(Pair.make(definer, "definition"));
    }

    @Override
    public String toString() {
      return toString(false);
    }

    @Override
    public Position position() {
      return pos;
    }

    private String checkReshape() {
      boolean first = true;
      int n = 0;
      String shape = "";
      for (Dimension<?> d : from) {
        if (d instanceof CompoundDim) {
          for (Dimension<?> dd : ((CompoundDim) d).value()) {
            shape = shape + (!first ? ", " : "") + dd.value();
            first = false;
          }
          n += ((CompoundDim) d).value().size();
        } else {
          shape = shape + (!first ? ", " : "") + ((d instanceof SymbolicDim) ? "-1" : d.value());
          first = false;
          n++;
        }
      }
      if (n == dims + 1) {
        n++;
        shape = shape + ", 1";
      }
      if (n == dims + 2) {
        try {
          SourceBuffer s = new SourceBuffer(pos);
          return "tf.reshape(" + s.toString() + ", [" + shape + "])";
        } catch (IOException e) {
          e.printStackTrace();
          return null;
        }
      } else {
        return null;
      }
    }

    @Override
    public String toString(boolean useMarkdown) {
      String msg =
          "Bad type to convolve "
              + from.toCString(useMarkdown)
              + ", needs "
              + (dims + 2)
              + " dimensions";
      String newDims = checkReshape();
      if (newDims != null) {
        msg = msg + " (possible fix: " + newDims + ")";
      }
      return msg;
    }

    @Override
    public DiagnosticSeverity severity() {
      return DiagnosticSeverity.Error;
    }

    @Override
    public String source() {
      return "Ariadne";
    }
  }

  /**
   * A transfer function that pins its destination's state to empty and fixed &mdash; blocks any
   * predecessor contributions from propagating into {@code lhs}. Used for PointsToSetVariables that
   * are semantically non-tensor (e.g., the integer index of {@code enumerate()}'s first tuple
   * field) but whose PA-graph predecessors would otherwise leak tensor types in via the assignment
   * graph's union semantics. See wala/ML#409.
   */
  static final class DropOp extends UnaryOperator<TensorVariable> {
    static final DropOp INSTANCE = new DropOp();

    private DropOp() {}

    @Override
    public byte evaluate(TensorVariable lhs, TensorVariable rhs) {
      boolean changed = false;
      if (lhs != null) {
        if (lhs.state != null && !lhs.state.isEmpty()) {
          lhs.state.clear();
          changed = true;
        }
        // A dropped destination is semantically non-tensor; leaked origin evidence goes with the
        // leaked types (wala/ML#724).
        if (!lhs.origins.isEmpty()) {
          lhs.origins.clear();
          changed = true;
        }
      }
      return changed ? CHANGED_AND_FIXED : NOT_CHANGED_AND_FIXED;
    }

    @Override
    public int hashCode() {
      return 0x7E9D50FF; // arbitrary constant; this is a singleton
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof DropOp;
    }

    @Override
    public String toString() {
      return "drop tensor types (enumerate first-field / non-tensor pin)";
    }
  }

  /**
   * A transfer function for parameter destinations (wala/ML#726): tensor types flow through
   * unchanged, but the origins union is skipped, so a parameter keeps its seeded {@link
   * TensorOrigin#PARAMETER} instead of inheriting its call sites' origins. Mirrors the plain node
   * transfer's state handling minus the provenance; in particular a ⊤-state (unknown tensor)
   * predecessor contributes nothing here, since its only contribution in the plain transfer is its
   * origins.
   */
  static final class ParameterBarrierOp extends UnaryOperator<TensorVariable> {
    static final ParameterBarrierOp INSTANCE = new ParameterBarrierOp();

    private ParameterBarrierOp() {}

    @Override
    public byte evaluate(TensorVariable lhs, TensorVariable rhs) {
      if (lhs == null || rhs == null || rhs.state == null) return NOT_CHANGED;
      if (lhs.state == null) {
        lhs.state = HashSetFactory.make(rhs.state);
        return CHANGED;
      }
      return lhs.state.addAll(rhs.state) ? CHANGED : NOT_CHANGED;
    }

    @Override
    public int hashCode() {
      return 0x726BA44E; // arbitrary constant; this is a singleton
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ParameterBarrierOp;
    }

    @Override
    public String toString() {
      return "propagate tensor types, block origin inflow (parameter boundary)";
    }
  }

  /**
   * A transfer function for iteration-product destinations (wala/ML#729): tensor types flow through
   * unchanged and non-parameter origin evidence flows with them, but {@link TensorOrigin#PARAMETER}
   * is filtered from the inflow. Iterating a symbolic tensor raises under {@code tf.function}
   * tracing (the iteration protocol is Python-level, not a traceable op), so a value iterated out
   * of a parameter is an eager-only product of the fed data: its provenance comes from its own
   * seed's creator walk, which reaches the caller-side producer, and the PA's aliasing of iteration
   * results with their iterables must not pull the parameter constant across.
   */
  static final class ParameterOriginFilterOp extends UnaryOperator<TensorVariable> {
    static final ParameterOriginFilterOp INSTANCE = new ParameterOriginFilterOp();

    private ParameterOriginFilterOp() {}

    @Override
    public byte evaluate(TensorVariable lhs, TensorVariable rhs) {
      if (lhs == null || rhs == null) return NOT_CHANGED;
      boolean changed = false;
      if (rhs.state != null) {
        if (lhs.state == null) {
          lhs.state = HashSetFactory.make(rhs.state);
          changed = true;
        } else changed |= lhs.state.addAll(rhs.state);
      }
      for (TensorOrigin origin : rhs.origins)
        if (origin != TensorOrigin.PARAMETER) changed |= lhs.origins.add(origin);
      return changed ? CHANGED : NOT_CHANGED;
    }

    @Override
    public int hashCode() {
      return 0x729F117E; // arbitrary constant; this is a singleton
    }

    @Override
    public boolean equals(Object o) {
      return o instanceof ParameterOriginFilterOp;
    }

    @Override
    public String toString() {
      return "propagate tensor types, filter the parameter origin (iteration product)";
    }
  }

  private static IKilldallFramework<PointsToSetVariable, TensorVariable> createProblem(
      Graph<PointsToSetVariable> G,
      Map<PointsToSetVariable, TensorType> reshapeNodes,
      Map<PointsToSetVariable, TensorType> set_shapes,
      Map<PointsToSetVariable, List<Dimension<?>>> refinements,
      Set<PointsToSetVariable> conv2ds,
      Set<PointsToSetVariable> conv3ds,
      Set<PointsToSetVariable> drops,
      Set<PointsToSetVariable> parameters,
      Set<PointsToSetVariable> iterationProducts,
      Map<PointerKey, AnalysisError> errorLog) {
    return new IKilldallFramework<PointsToSetVariable, TensorVariable>() {

      @Override
      public Graph<PointsToSetVariable> getFlowGraph() {
        return G;
      }

      @Override
      public ITransferFunctionProvider<PointsToSetVariable, TensorVariable>
          getTransferFunctionProvider() {
        return new ITransferFunctionProvider<PointsToSetVariable, TensorVariable>() {

          private Position getTargetPos(PointerKey pk) {
            if (pk instanceof LocalPointerKey) {
              LocalPointerKey lpk = (LocalPointerKey) pk;
              DefUse du = lpk.getNode().getDU();
              SSAInstruction inst = du.getDef(lpk.getValueNumber());
              if (lpk.getNode().getMethod() instanceof AstMethod) {
                return ((AstMethod) lpk.getNode().getMethod())
                    .debugInfo()
                    .getOperandPosition(inst.iIndex(), 1);
              }
            }

            return null;
          }

          private Position getTargetDef(PointerKey pk) {
            if (pk instanceof LocalPointerKey) {
              LocalPointerKey lpk = (LocalPointerKey) pk;
              DefUse du = lpk.getNode().getDU();
              SSAInstruction inst = du.getDef(lpk.getValueNumber());
              if (lpk.getNode().getMethod() instanceof AstMethod) {
                SSAInstruction def = du.getDef(inst.getUse(1));
                // A synthesized definition (e.g. a constructor-injected trampoline wire or a phi)
                // has a negative instruction index and no source position.
                if (def != null && def.iIndex() >= 0) {
                  return ((AstMethod) lpk.getNode().getMethod())
                      .debugInfo()
                      .getInstructionPosition(def.iIndex());
                }
              }
            }

            return null;
          }

          final class SetShapeOp extends UnaryOperator<TensorVariable> {
            private final TensorType setShapeTo;

            public SetShapeOp(TensorType reshapeTo) {
              this.setShapeTo = reshapeTo;
            }

            @Override
            public byte evaluate(TensorVariable lhs, TensorVariable rhs) {
              // https://github.com/wala/ML/issues/509: `x.set_shape(s)` is a user-supplied shape
              // assertion that should
              // OVERRIDE any per-op-generator init seed on the receiver, not union with it. The
              // previous `add`-only semantics caused the cast result's Cast-generator-seeded
              // `(?, dtype)` to leak into the post-set_shape state alongside the asserted shape.
              // Clear-then-add mirrors `DropOp`'s clear-state pattern with CHANGED_AND_FIXED.
              boolean changed = false;
              if (!(lhs.state.size() == 1 && lhs.state.contains(setShapeTo))) {
                lhs.state.clear();
                lhs.state.add(setShapeTo);
                changed = true;
              }
              // Pinning replaces the SHAPE, not the provenance: the destination's value is still
              // produced by whatever produced the incoming flow (a `set_shape` receiver keeps its
              // pre-assertion origins; a subscript result routed here by the wala/ML#405 reroute
              // keeps its container's), so origins union from the predecessor as everywhere else
              // (wala/ML#724).
              if (rhs != null) changed |= lhs.origins.addAll(rhs.origins);
              return changed ? CHANGED_AND_FIXED : NOT_CHANGED_AND_FIXED;
            }

            @Override
            public int hashCode() {
              return setShapeTo.hashCode();
            }

            @Override
            public boolean equals(Object o) {
              return this == o
                  || ((o instanceof ReshapeOp) && setShapeTo.equals(((ReshapeOp) o).reshapeTo));
            }

            @Override
            public String toString() {
              return "set shape to " + setShapeTo;
            }
          }

          /**
           * A transfer function for einsum-operand destinations (wala/ML#704): each incoming tensor
           * type is refined against the shape the einsum equation proves for the operand. An
           * unknown-rank (⊤-shape) member takes the proven shape outright, keeping its dtype and
           * layout; a member of the proven rank fills each non-numeric axis from the constraint's
           * known one. A member that contradicts the constraint (a different rank) passes through
           * unchanged: such a value fails at the einsum call at runtime, and the analysis reports
           * what flows rather than fabricating agreement. Unlike {@code SetShapeOp} this is a
           * monotone per-member refinement, not a pin: predecessors keep contributing, and concrete
           * incoming state is never discarded.
           *
           * <p>Origin handling matches the transfer this op displaces on its destination: a
           * parameter destination blocks origin inflow ({@code ParameterBarrierOp}, wala/ML#726),
           * an iteration-product destination filters {@link TensorOrigin#PARAMETER} ({@code
           * ParameterOriginFilterOp}, wala/ML#729), and any other destination unions origins as the
           * plain transfer does.
           */
          final class RefineShapeOp extends UnaryOperator<TensorVariable> {
            private final List<Dimension<?>> constraint;
            private final PointsToSetVariable destination;

            public RefineShapeOp(List<Dimension<?>> constraint, PointsToSetVariable destination) {
              this.constraint = constraint;
              this.destination = destination;
            }

            @Override
            public byte evaluate(TensorVariable lhs, TensorVariable rhs) {
              if (lhs == null || rhs == null) return NOT_CHANGED;
              boolean changed = false;
              if (rhs.state != null) {
                if (lhs.state == null) lhs.state = HashSetFactory.make();
                for (TensorType t : rhs.state) changed |= lhs.state.add(this.refine(t));
              }
              if (!parameters.contains(this.destination)) {
                for (TensorOrigin origin : rhs.origins)
                  if (origin != TensorOrigin.PARAMETER
                      || !iterationProducts.contains(this.destination))
                    changed |= lhs.origins.add(origin);
              }
              return changed ? CHANGED : NOT_CHANGED;
            }

            /**
             * Refines one incoming member against the proven operand shape.
             *
             * @param t The incoming tensor type.
             * @return The refined type: the proven shape (with {@code t}'s dtype and layout) when
             *     {@code t}'s rank is unknown, {@code t} with each non-numeric axis filled from the
             *     constraint's known one when the ranks agree, and {@code t} itself otherwise.
             */
            private TensorType refine(TensorType t) {
              List<Dimension<?>> dims = t.getDims();
              if (dims == null) return TensorType.of(t.getDType(), this.constraint, t.layout());
              if (dims.size() != this.constraint.size()) return t;
              List<Dimension<?>> merged = new ArrayList<>(dims.size());
              boolean refined = false;
              for (int i = 0; i < dims.size(); i++) {
                Dimension<?> have = dims.get(i);
                Dimension<?> want = this.constraint.get(i);
                if (!(have instanceof NumericDim) && want instanceof NumericDim) {
                  merged.add(want);
                  refined = true;
                } else merged.add(have);
              }
              return refined ? TensorType.of(t.getDType(), merged, t.layout()) : t;
            }

            @Override
            public int hashCode() {
              return this.constraint.hashCode() * 31 + this.destination.hashCode();
            }

            @Override
            public boolean equals(Object o) {
              return o instanceof RefineShapeOp
                  && this.constraint.equals(((RefineShapeOp) o).constraint)
                  && this.destination.equals(((RefineShapeOp) o).destination);
            }

            @Override
            public String toString() {
              return "refine shape against einsum operand constraint " + this.constraint;
            }
          }

          final class ConvOp extends UnaryOperator<TensorVariable> {
            private final PointsToSetVariable v;
            private final int dimensions;

            public ConvOp(int dimensions, PointsToSetVariable v) {
              this.v = v;
              this.dimensions = dimensions;
            }

            @Override
            public byte evaluate(TensorVariable lhs, TensorVariable rhs) {
              boolean changed = false;
              if (rhs != null && rhs.state != null) {
                for (TensorType t : rhs.state) {
                  int dims = 0;
                  for (Dimension<?> d : t) {
                    if (d != null) {
                      dims++;
                    }
                  }
                  if (dims == dimensions + 2) {
                    changed |= lhs.state.add(t);
                  } else {
                    Position pos = getTargetPos(v.getPointerKey());
                    errorLog.put(
                        v.getPointerKey(),
                        new ConvError(t, dimensions, pos, getTargetDef(v.getPointerKey())));
                  }
                }
                // A convolution is a TensorFlow operation; its result is TensorFlow-origin
                // whatever produced the input (wala/ML#724).
                changed |= lhs.origins.add(TensorOrigin.TENSORFLOW);
              }
              return changed ? CHANGED_AND_FIXED : NOT_CHANGED;
            }

            @Override
            public int hashCode() {
              return v.hashCode();
            }

            @Override
            public boolean equals(Object o) {
              return (o instanceof ConvOp) && ((ConvOp) o).v.equals(v);
            }

            @Override
            public String toString() {
              return "conv at " + v;
            }
          }

          final class ReshapeOp extends UnaryOperator<TensorVariable> {
            private final TensorType reshapeTo;
            private final PointsToSetVariable v;

            public ReshapeOp(TensorType reshapeTo, PointsToSetVariable v) {
              this.v = v;
              this.reshapeTo = reshapeTo;
            }

            @Override
            public byte evaluate(TensorVariable lhs, TensorVariable rhs) {
              boolean changed = false;
              int ssz = reshapeTo.symbolicDims();
              int csz = reshapeTo.concreteSize();
              if (rhs != null && rhs.state != null) {
                for (TensorType t : rhs.state) {
                  TensorType newType = reshapeTo;
                  // If there is exactly one dynamic dimension (symbolic), try to resolve it by
                  // comparing the total size of the input tensor with the concrete size of the
                  // target shape.
                  if (ssz == 1 && t.symbolicDims() == 0 && t.concreteSize() != -1) {
                    int totalSize = t.concreteSize();
                    int partialSize = 1;
                    for (Dimension<?> d : reshapeTo.getDims()) {
                      if (d instanceof NumericDim) {
                        partialSize *= ((NumericDim) d).value();
                      }
                    }

                    if (partialSize > 0 && totalSize % partialSize == 0) {
                      int missingDim = totalSize / partialSize;
                      List<Dimension<?>> newDims = new ArrayList<>();
                      for (Dimension<?> d : reshapeTo.getDims()) {
                        if (d instanceof SymbolicDim) {
                          newDims.add(new NumericDim(missingDim));
                        } else {
                          newDims.add(d);
                        }
                      }
                      // Use the source tensor's cell type to ensure precision in the reshaped type.
                      newType = new TensorType(t.getCellType(), newDims);
                    } else {
                      newType = new TensorType(t.getCellType(), reshapeTo.getDims());
                    }
                  } else {
                    newType = new TensorType(t.getCellType(), reshapeTo.getDims());
                  }

                  // Process the reshape normally regardless of whether there is a shape mismatch so
                  // that tensor type propagation continues. The shape may be inaccurate, but users
                  // can inspect the error messages to find out about it. See
                  // https://github.com/wala/ML/issues/195.
                  if (lhs.state == null) {
                    lhs.state = HashSetFactory.make();
                  }
                  changed |= lhs.state.add(newType);

                  if (t.symbolicDims() != ssz || t.concreteSize() != csz) {
                    Position pos = getTargetPos(v.getPointerKey());
                    assert pos != null;
                    errorLog.put(
                        v.getPointerKey(),
                        new ReshapeError(t, reshapeTo, pos, getTargetDef(v.getPointerKey())));
                  }
                }
                // `tf.reshape` is a TensorFlow operation; its result is TensorFlow-origin
                // whatever produced the input (wala/ML#724).
                changed |= lhs.origins.add(TensorOrigin.TENSORFLOW);
              }
              return changed ? CHANGED_AND_FIXED : NOT_CHANGED;
            }

            @Override
            public int hashCode() {
              return reshapeTo.hashCode();
            }

            @Override
            public boolean equals(Object o) {
              return this == o
                  || ((o instanceof ReshapeOp) && reshapeTo.equals(((ReshapeOp) o).reshapeTo));
            }

            @Override
            public String toString() {
              return "reshape to " + reshapeTo;
            }
          }

          private final UnaryOperator<TensorVariable> nodeOp =
              new UnaryOperator<TensorVariable>() {
                @Override
                public byte evaluate(TensorVariable lhs, TensorVariable rhs) {
                  // The solver constructs every node variable, so a null endpoint has nothing to
                  // update; the guard also keeps the null tests below honest (the previous
                  // `lhs == null || lhs.state == null` guard dereferenced `lhs` in its own arm).
                  if (lhs == null || rhs == null) return NOT_CHANGED;

                  if (rhs.state != null) {
                    if (lhs.state == null) {
                      lhs.copyState(rhs);
                      return CHANGED;
                    }
                    boolean changed = lhs.state.addAll(rhs.state);
                    changed |= lhs.origins.addAll(rhs.origins);
                    return changed ? CHANGED : NOT_CHANGED;
                  }

                  // A null-state (unknown tensor, ⊤) predecessor contributes no types, but its
                  // producing library is still evidence and must flow: provenance matters most
                  // exactly when the shape is unknown (wala/ML#724).
                  return lhs.origins.addAll(rhs.origins) ? CHANGED : NOT_CHANGED;
                }

                @Override
                public int hashCode() {
                  return 817504253;
                }

                @Override
                public boolean equals(Object o) {
                  return o == this;
                }

                @Override
                public String toString() {
                  return "propagate node tensor types";
                }
              };

          @Override
          public UnaryOperator<TensorVariable> getNodeTransferFunction(PointsToSetVariable node) {
            if (drops.contains(node)) {
              return DropOp.INSTANCE;
            } else if (reshapeNodes.containsKey(node)) {
              return new ReshapeOp(reshapeNodes.get(node), node);
            } else if (conv2ds.contains(node)) {
              return new ConvOp(2, node);
            } else if (conv3ds.contains(node)) {
              return new ConvOp(3, node);
            } else if (parameters.contains(node)) {
              return ParameterBarrierOp.INSTANCE;
            } else if (iterationProducts.contains(node)) {
              return ParameterOriginFilterOp.INSTANCE;
            } else {
              return nodeOp;
            }
          }

          @Override
          public boolean hasNodeTransferFunctions() {
            return true;
          }

          @Override
          public UnaryOperator<TensorVariable> getEdgeTransferFunction(
              PointsToSetVariable src, PointsToSetVariable dst) {
            if (drops.contains(dst)) {
              return DropOp.INSTANCE;
            } else if (set_shapes.containsKey(dst)) {
              return new SetShapeOp(set_shapes.get(dst));
            } else if (refinements.containsKey(dst)) {
              return new RefineShapeOp(refinements.get(dst), dst);
            } else if (parameters.contains(dst)) {
              return ParameterBarrierOp.INSTANCE;
            } else if (iterationProducts.contains(dst)) {
              return ParameterOriginFilterOp.INSTANCE;
            } else {
              return nodeOp;
            }
          }

          @Override
          public boolean hasEdgeTransferFunctions() {
            return true;
          }

          @Override
          public AbstractMeetOperator<TensorVariable> getMeetOperator() {
            return new AbstractMeetOperator<TensorVariable>() {

              @Override
              public byte evaluate(TensorVariable lhs, TensorVariable[] rhs) {
                boolean changed = false;
                for (TensorVariable r : rhs) {
                  changed |= lhs.state.addAll(r.state);
                  changed |= lhs.origins.addAll(r.origins);
                }
                return changed ? CHANGED : NOT_CHANGED;
              }

              @Override
              public int hashCode() {
                return 413158523;
              }

              @Override
              public boolean equals(Object o) {
                return this == o;
              }

              @Override
              public String toString() {
                return "Tensor types set union";
              }
            };
          }
        };
      }
    };
  }

  private final Map<PointsToSetVariable, Set<TensorType>> init;

  private final Map<PointsToSetVariable, Set<TensorOrigin>> initOrigins;

  /**
   * Constructs the tensor type dataflow analysis over the PA assignment graph.
   *
   * @param G The dataflow graph (the PA assignment graph including implicit constraints).
   * @param init The per-source seeded tensor types; a {@code null} value seeds an unknown tensor.
   * @param initOrigins The per-source seeded producing libraries (wala/ML#724), unioned along the
   *     same edges as the types.
   * @param reshapeTypes Destinations pinned by an explicit reshape target shape.
   * @param set_shapes Destinations pinned by an {@code x.set_shape(s)} assertion (wala/ML#509).
   * @param refinements Einsum-operand destinations refined against the shape the einsum equation
   *     proves for them (wala/ML#704); incoming members are refined, never discarded.
   * @param conv2ds Destinations of 2D convolutions, rank-checked by {@code ConvOp}.
   * @param conv3ds Destinations of 3D convolutions, rank-checked by {@code ConvOp}.
   * @param drops Destinations pinned to empty-and-fixed (wala/ML#409).
   * @param parameters Parameter destinations, whose types flow normally but whose origins are
   *     pinned to their {@link TensorOrigin#PARAMETER} seed by blocking caller-side origin inflow
   *     (wala/ML#726).
   * @param iterationProducts Iteration-product destinations, whose types flow normally but whose
   *     origin inflow drops {@link TensorOrigin#PARAMETER} (wala/ML#729).
   * @param errorLog The sink for shape-mismatch diagnostics.
   */
  public TensorTypeAnalysis(
      Graph<PointsToSetVariable> G,
      Map<PointsToSetVariable, Set<TensorType>> init,
      Map<PointsToSetVariable, Set<TensorOrigin>> initOrigins,
      Map<PointsToSetVariable, TensorType> reshapeTypes,
      Map<PointsToSetVariable, TensorType> set_shapes,
      Map<PointsToSetVariable, List<Dimension<?>>> refinements,
      Set<PointsToSetVariable> conv2ds,
      Set<PointsToSetVariable> conv3ds,
      Set<PointsToSetVariable> drops,
      Set<PointsToSetVariable> parameters,
      Set<PointsToSetVariable> iterationProducts,
      Map<PointerKey, AnalysisError> errorLog) {
    super(
        createProblem(
            G,
            reshapeTypes,
            set_shapes,
            refinements,
            conv2ds,
            conv3ds,
            drops,
            parameters,
            iterationProducts,
            errorLog));
    this.init = init;
    this.initOrigins = initOrigins;
  }

  @Override
  protected TensorVariable makeNodeVariable(PointsToSetVariable n, boolean IN) {
    return new TensorVariable();
  }

  @Override
  protected TensorVariable makeEdgeVariable(PointsToSetVariable src, PointsToSetVariable dst) {
    return new TensorVariable();
  }

  @Override
  protected TensorVariable[] makeStmtRHS(int size) {
    return new TensorVariable[size];
  }

  @Override
  protected void initializeVariables() {
    super.initializeVariables();
    for (PointsToSetVariable src : init.keySet()) {
      Set<TensorType> types = init.get(src);
      if (types != null) {
        getOut(src).state.addAll(types);
      } else {
        getOut(src).state = null; // unknown tensor — distinguishable from empty (not-a-tensor)
      }
    }
    for (PointsToSetVariable src : initOrigins.keySet())
      getOut(src).origins.addAll(initOrigins.get(src));
  }

  public String toString() {
    StringBuffer sb = new StringBuffer("answer:\n");
    for (PointsToSetVariable var : getProblem().getFlowGraph()) {
      if (getOut(var) != null && getOut(var).state != null && !getOut(var).state.isEmpty()) {
        sb.append(var.getPointerKey()).append(getOut(var)).append("\n");
      }
    }
    return sb.toString();
  }

  @Override
  public Iterator<Pair<PointerKey, TensorVariable>> iterator() {
    Set<Pair<PointerKey, TensorVariable>> x = HashSetFactory.make();
    for (PointsToSetVariable var : getProblem().getFlowGraph()) {
      if (getOut(var) != null && getOut(var).state != null && !getOut(var).state.isEmpty()) {
        x.add(Pair.make(var.getPointerKey(), getOut(var)));
      }
    }
    return x.iterator();
  }
}
