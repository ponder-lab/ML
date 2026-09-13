package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.NumpyTypes;
import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.cast.python.ml.types.TensorType.Dimension;
import com.ibm.wala.cast.python.ml.types.TensorType.NumericDim;
import com.ibm.wala.cast.python.ml.types.TensorType.UnresolvedDim;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.ssa.PythonPropertyWrite;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import com.ibm.wala.shrike.shrikeBT.IBinaryOpInstruction;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.ssa.SSABinaryOpInstruction;
import com.ibm.wala.ssa.SSAInstruction;
import com.ibm.wala.ssa.SSANewInstruction;
import com.ibm.wala.ssa.SSAPutInstruction;
import com.ibm.wala.ssa.SymbolTable;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.logging.Logger;

/**
 * Generator for {@code np.pad(array, pad_width, mode, ...)} (wala/ML#909): the rank is the input's,
 * and every axis grows by its {@code before} and {@code after} widths. The mode never affects the
 * shape.
 *
 * <p>The extents are folded as linear terms over the program's own values rather than as numbers,
 * because a padding is how a program fixes a length it does not otherwise know: {@code np.pad(seq,
 * (0, total - n))} over a {@code seq} that is {@code n} long is {@code total} long, and {@code n}
 * may be a runtime value. Dimension arithmetic resolves values and so cannot see that the two
 * occurrences of {@code n} cancel; this generator chases the input to the producer that fixed its
 * length ({@code np.arange} bounds, a sized random draw's {@code size}, a {@code zeros}/{@code
 * ones} shape, through an elementwise operation with a scalar) and folds the producer's terms and
 * the widths' terms together. An axis whose sum is not a constant is {@link UnresolvedDim}, a fixed
 * runtime integer the analysis could not compute (wala/ML#721).
 *
 * <p>The term arithmetic is deliberately private to this generator and never enters the dimension
 * lattice: what reaches the lattice is a {@link NumericDim} or an {@link UnresolvedDim}, exactly as
 * today. Widening its use to other generators reopens the question wala/ML#909 poses about term
 * identity in the lattice, which this generator answers only for its own operands.
 *
 * @see <a href="https://numpy.org/doc/stable/reference/generated/numpy.pad.html">numpy.pad</a>
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class NpPad extends TensorGenerator {

  private static final Logger LOGGER = Logger.getLogger(NpPad.class.getName());

  /** Hops the producer chase follows through elementwise operations before giving up. */
  private static final int CHASE_DEPTH = 6;

  /** The sized random draws whose {@code size} argument fixes their shape. */
  private static final List<TypeReference> SIZED_DRAWS =
      List.of(
          NumpyTypes.RANDOM_RANDINT.getDeclaringClass(),
          NumpyTypes.RANDOM_NORMAL.getDeclaringClass(),
          NumpyTypes.RANDOM_UNIFORM.getDeclaringClass());

  /** The constructors whose first argument is the shape. */
  private static final List<TypeReference> SHAPED_CONSTRUCTORS =
      List.of(NumpyTypes.ZEROS.getDeclaringClass(), NumpyTypes.ONES.getDeclaringClass());

  public NpPad(PointsToSetVariable source) {
    super(source);
  }

  /**
   * Constructs anchored to a manual node.
   *
   * @param node The {@link CGNode} for the synthetic {@code do()} method.
   */
  public NpPad(CGNode node) {
    super(node);
  }

  @Override
  protected Set<List<Dimension<?>>> getDefaultShapes(PropagationCallGraphBuilder builder) {
    Set<List<Dimension<?>>> ret = HashSetFactory.make();
    for (Pair<CGNode, PythonInvokeInstruction> callerInvoke : this.callerInvokes(builder)) {
      CGNode caller = callerInvoke.fst;
      PythonInvokeInstruction call = callerInvoke.snd;
      int arrayVn = argumentValueNumber(call, 1, "array");
      int widthVn = argumentValueNumber(call, 2, "pad_width");
      if (arrayVn < 0) continue;

      // The input's per-axis extents as terms: from the producer chase when it reaches one, else
      // from the input's typed shape (constants become constant terms, anything else unresolvable).
      Set<List<LinearTerm>> inputs = HashSetFactory.make();
      List<LinearTerm> chased = inputExtentTerms(builder, caller, arrayVn, CHASE_DEPTH);
      if (chased != null) inputs.add(chased);
      else {
        Set<List<Dimension<?>>> shapes = this.getShapes(builder, caller, arrayVn);
        if (shapes == null) continue;
        for (List<Dimension<?>> shape : shapes) {
          List<LinearTerm> terms = new ArrayList<>();
          for (Dimension<?> dim : shape)
            terms.add(
                dim instanceof NumericDim ? LinearTerm.constant((Integer) dim.value()) : null);
          inputs.add(terms);
        }
      }

      for (List<LinearTerm> input : inputs) {
        List<LinearTerm[]> widths = widths(builder, caller, widthVn, input.size());
        List<Dimension<?>> out = new ArrayList<>();
        for (int axis = 0; axis < input.size(); axis++) {
          LinearTerm in = input.get(axis);
          LinearTerm[] w = widths == null ? null : widths.get(axis);
          LinearTerm total =
              in == null || w == null || w[0] == null || w[1] == null
                  ? null
                  : in.plus(w[0]).plus(w[1]);
          out.add(
              total != null && total.isConstant()
                  ? new NumericDim((int) total.constant())
                  : UnresolvedDim.INSTANCE);
        }
        ret.add(out);
      }
    }
    LOGGER.fine(() -> "np.pad resolved to " + ret + ".");
    return ret.isEmpty() ? null : ret;
  }

  /**
   * The call sites this generator reads its arguments from: the anchoring invoke when the source is
   * anchored in a caller frame, else every reachable invoke of the anchored node.
   */
  private List<Pair<CGNode, PythonInvokeInstruction>> callerInvokes(
      PropagationCallGraphBuilder builder) {
    List<Pair<CGNode, PythonInvokeInstruction>> ret = new ArrayList<>();
    PythonInvokeInstruction own = this.getInvokeInstruction();
    if (own != null) ret.add(Pair.make(this.getNode(), own));
    else
      for (Pair<CGNode, SSAAbstractInvokeInstruction> callerInvoke :
          getCallerInvokes(builder, this.getNode()))
        if (callerInvoke.snd instanceof PythonInvokeInstruction)
          ret.add(Pair.make(callerInvoke.fst, (PythonInvokeInstruction) callerInvoke.snd));
    return ret;
  }

  /**
   * An argument's value number by keyword, else by position ({@code 0} is the callee).
   *
   * @return The value number, or {@code -1} when the call does not supply it.
   */
  private static int argumentValueNumber(PythonInvokeInstruction call, int position, String name) {
    if (call.getKeywords().contains(name)) return call.getUse(name);
    return call.getNumberOfPositionalParameters() > position ? call.getUse(position) : -1;
  }

  /**
   * Chases the padded value to the producer that fixed its extents and returns them as terms:
   * {@code np.arange}'s {@code stop - start} (unit step), a sized draw's {@code size}, a shaped
   * constructor's {@code shape}, each reached through elementwise operations with a scalar.
   *
   * @return The per-axis terms ({@code null} entries for an axis no term expresses), or {@code
   *     null} when the chase reaches no producer it can read.
   */
  private static List<LinearTerm> inputExtentTerms(
      PropagationCallGraphBuilder builder, CGNode node, int vn, int depth) {
    if (depth <= 0 || vn <= 0 || node.getDU() == null || node.getIR() == null) return null;
    SymbolTable st = node.getIR().getSymbolTable();
    SSAInstruction def = node.getDU().getDef(vn);
    if (def instanceof SSABinaryOpInstruction) {
      // An elementwise operation with a scalar keeps the array operand's shape.
      SSABinaryOpInstruction binOp = (SSABinaryOpInstruction) def;
      if (st.isNumberConstant(binOp.getUse(1)))
        return inputExtentTerms(builder, node, binOp.getUse(0), depth - 1);
      if (st.isNumberConstant(binOp.getUse(0)))
        return inputExtentTerms(builder, node, binOp.getUse(1), depth - 1);
      return null;
    }
    if (!(def instanceof PythonInvokeInstruction)) return null;
    PythonInvokeInstruction call = (PythonInvokeInstruction) def;
    if (calleeIs(builder, node, call, List.of(NumpyTypes.ARANGE.getDeclaringClass()))) {
      int[] bounds = NpArange.boundValueNumbers(call);
      if (bounds[1] < 0) return null;
      if (bounds[2] >= 0) {
        LinearTerm step = LinearTerm.resolve(builder, node, bounds[2], depth);
        if (!step.isConstant() || step.constant() != 1) return null;
      }
      LinearTerm start =
          bounds[0] < 0
              ? LinearTerm.constant(0)
              : LinearTerm.resolve(builder, node, bounds[0], depth);
      LinearTerm stop = LinearTerm.resolve(builder, node, bounds[1], depth);
      return List.of(stop.minus(start));
    }
    if (calleeIs(builder, node, call, SIZED_DRAWS))
      return shapeTerms(builder, node, argumentValueNumber(call, 3, "size"), depth);
    if (calleeIs(builder, node, call, SHAPED_CONSTRUCTORS))
      return shapeTerms(builder, node, argumentValueNumber(call, 1, "shape"), depth);
    return null;
  }

  /** A shape argument, an integer or a tuple of them, as per-axis terms. */
  private static List<LinearTerm> shapeTerms(
      PropagationCallGraphBuilder builder, CGNode node, int vn, int depth) {
    if (vn < 0) return null;
    Map<Integer, Integer> elements = tupleElements(node, vn);
    if (elements == null) return List.of(LinearTerm.resolve(builder, node, vn, depth));
    List<LinearTerm> ret = new ArrayList<>();
    for (int i = 0; i < elements.size(); i++) {
      Integer element = elements.get(i);
      if (element == null) return null;
      ret.add(LinearTerm.resolve(builder, node, element, depth));
    }
    return ret;
  }

  /**
   * The {@code pad_width} argument as per-axis {@code (before, after)} terms: an integer pads every
   * side of every axis, a {@code (before, after)} pair pads every axis alike, and a sequence of
   * pairs pads each axis on its own.
   *
   * @return The per-axis widths, or {@code null} when the argument is absent or not a form this
   *     generator reads.
   */
  private static List<LinearTerm[]> widths(
      PropagationCallGraphBuilder builder, CGNode node, int vn, int rank) {
    if (vn < 0) return null;
    Map<Integer, Integer> elements = tupleElements(node, vn);
    List<LinearTerm[]> ret = new ArrayList<>();
    if (elements == null) {
      LinearTerm both = LinearTerm.resolve(builder, node, vn, CHASE_DEPTH);
      for (int axis = 0; axis < rank; axis++) ret.add(new LinearTerm[] {both, both});
      return ret;
    }
    Integer first = elements.get(0);
    if (first == null) return null;
    if (elements.size() == 1 && tupleElements(node, first) == null) {
      LinearTerm both = LinearTerm.resolve(builder, node, first, CHASE_DEPTH);
      for (int axis = 0; axis < rank; axis++) ret.add(new LinearTerm[] {both, both});
      return ret;
    }
    Integer second = elements.get(1);
    if (elements.size() == 2
        && second != null
        && tupleElements(node, first) == null
        && tupleElements(node, second) == null) {
      LinearTerm before = LinearTerm.resolve(builder, node, first, CHASE_DEPTH);
      LinearTerm after = LinearTerm.resolve(builder, node, second, CHASE_DEPTH);
      for (int axis = 0; axis < rank; axis++) ret.add(new LinearTerm[] {before, after});
      return ret;
    }
    if (elements.size() != rank) return null;
    for (int axis = 0; axis < rank; axis++) {
      Integer pairVn = elements.get(axis);
      Map<Integer, Integer> pair = pairVn == null ? null : tupleElements(node, pairVn);
      if (pair == null || pair.size() != 2 || pair.get(0) == null || pair.get(1) == null)
        return null;
      ret.add(
          new LinearTerm[] {
            LinearTerm.resolve(builder, node, pair.get(0), CHASE_DEPTH),
            LinearTerm.resolve(builder, node, pair.get(1), CHASE_DEPTH)
          });
    }
    return ret;
  }

  /**
   * The elements of a tuple or list literal allocated in the node, by index, read from the field
   * writes on the allocation.
   *
   * @return The index-to-value map, or {@code null} when the value is not such a literal.
   */
  private static Map<Integer, Integer> tupleElements(CGNode node, int vn) {
    if (vn <= 0 || node.getDU() == null) return null;
    SSAInstruction def = node.getDU().getDef(vn);
    if (!(def instanceof SSANewInstruction)) return null;
    TypeReference type = ((SSANewInstruction) def).getConcreteType();
    if (!type.equals(PythonTypes.tuple) && !type.equals(PythonTypes.list)) return null;
    SymbolTable st = node.getIR().getSymbolTable();
    Map<Integer, Integer> ret = new TreeMap<>();
    for (Iterator<SSAInstruction> uses = node.getDU().getUses(vn); uses.hasNext(); ) {
      SSAInstruction use = uses.next();
      Integer index = null;
      int written;
      if (use instanceof PythonPropertyWrite) {
        PythonPropertyWrite write = (PythonPropertyWrite) use;
        if (write.getObjectRef() != vn) continue;
        int memberVn = write.getMemberRef();
        if (st.isNumberConstant(memberVn))
          index = ((Number) st.getConstantValue(memberVn)).intValue();
        else if (st.isStringConstant(memberVn)) {
          try {
            index = Integer.parseInt(st.getStringValue(memberVn));
          } catch (NumberFormatException e) {
            continue;
          }
        }
        written = write.getValue();
      } else if (use instanceof SSAPutInstruction && !((SSAPutInstruction) use).isStatic()) {
        SSAPutInstruction put = (SSAPutInstruction) use;
        if (put.getRef() != vn) continue;
        try {
          index = Integer.parseInt(put.getDeclaredField().getName().toString());
        } catch (NumberFormatException e) {
          continue;
        }
        written = put.getVal();
      } else continue;
      if (index != null) ret.put(index, written);
    }
    return ret;
  }

  /** Whether the invoke's callee value points to an instance of one of the given classes. */
  private static boolean calleeIs(
      PropagationCallGraphBuilder builder,
      CGNode node,
      PythonInvokeInstruction call,
      List<TypeReference> classes) {
    OrdinalSet<InstanceKey> pts =
        builder
            .getPointerAnalysis()
            .getPointsToSet(builder.getPointerKeyForLocal(node, call.getUse(0)));
    if (pts == null) return false;
    for (InstanceKey ik : pts)
      for (TypeReference type : classes)
        if (ik.concreteType().getReference().getName().equals(type.getName())) return true;
    return false;
  }

  @Override
  protected Set<DType> getDefaultDTypes(PropagationCallGraphBuilder builder) {
    // Padding keeps the input's dtype, read in the caller frame through the same invokes the
    // shapes come from; the generic argument fallback walks an SSA chain that can reach the integer
    // widths instead.
    Set<DType> ret = EnumSet.noneOf(DType.class);
    for (Pair<CGNode, PythonInvokeInstruction> callerInvoke : this.callerInvokes(builder)) {
      int arrayVn = argumentValueNumber(callerInvoke.snd, 1, "array");
      if (arrayVn < 0) continue;
      Set<DType> dtypes = this.getDTypes(builder, callerInvoke.fst, arrayVn);
      if (dtypes != null) ret.addAll(dtypes);
    }
    return ret.isEmpty() ? EnumSet.of(DType.UNKNOWN) : ret;
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
   * A linear expression over the program's own integer values: a constant plus coefficients on
   * atoms, an atom being an SSA value the resolver cannot fold further, identified by its node and
   * value number so that two reads of one value are one atom. Private to {@link NpPad} on purpose:
   * see the class comment.
   */
  private static final class LinearTerm {

    private final Map<Pair<CGNode, Integer>, Long> coefficients;

    private final long constant;

    private LinearTerm(Map<Pair<CGNode, Integer>, Long> coefficients, long constant) {
      Map<Pair<CGNode, Integer>, Long> normalized = HashMapFactory.make();
      for (Map.Entry<Pair<CGNode, Integer>, Long> entry : coefficients.entrySet())
        if (entry.getValue() != 0) normalized.put(entry.getKey(), entry.getValue());
      this.coefficients = normalized;
      this.constant = constant;
    }

    static LinearTerm constant(long value) {
      return new LinearTerm(HashMapFactory.make(), value);
    }

    static LinearTerm atom(CGNode node, int vn) {
      Map<Pair<CGNode, Integer>, Long> coefficients = HashMapFactory.make();
      coefficients.put(Pair.make(node, vn), 1L);
      return new LinearTerm(coefficients, 0);
    }

    boolean isConstant() {
      return this.coefficients.isEmpty();
    }

    long constant() {
      return this.constant;
    }

    LinearTerm plus(LinearTerm other) {
      Map<Pair<CGNode, Integer>, Long> sum = HashMapFactory.make(this.coefficients);
      for (Map.Entry<Pair<CGNode, Integer>, Long> entry : other.coefficients.entrySet())
        sum.merge(entry.getKey(), entry.getValue(), Long::sum);
      return new LinearTerm(sum, this.constant + other.constant);
    }

    LinearTerm minus(LinearTerm other) {
      return this.plus(other.scale(-1));
    }

    LinearTerm scale(long factor) {
      Map<Pair<CGNode, Integer>, Long> scaled = HashMapFactory.make();
      for (Map.Entry<Pair<CGNode, Integer>, Long> entry : this.coefficients.entrySet())
        scaled.put(entry.getKey(), entry.getValue() * factor);
      return new LinearTerm(scaled, this.constant * factor);
    }

    /**
     * Resolves a value to a term: an integer constant, a sum, difference or constant multiple of
     * terms, a value the flow-sensitive integer resolver settles, or else an atom.
     */
    static LinearTerm resolve(PropagationCallGraphBuilder builder, CGNode node, int vn, int depth) {
      SymbolTable st = node.getIR().getSymbolTable();
      if (st.isNumberConstant(vn)) {
        Number number = (Number) st.getConstantValue(vn);
        if (number.doubleValue() == Math.rint(number.doubleValue()))
          return constant(number.longValue());
        return atom(node, vn);
      }
      SSAInstruction def = node.getDU().getDef(vn);
      if (depth > 0 && def instanceof SSABinaryOpInstruction) {
        SSABinaryOpInstruction binOp = (SSABinaryOpInstruction) def;
        LinearTerm left = resolve(builder, node, binOp.getUse(0), depth - 1);
        LinearTerm right = resolve(builder, node, binOp.getUse(1), depth - 1);
        if (binOp.getOperator() == IBinaryOpInstruction.Operator.ADD) return left.plus(right);
        if (binOp.getOperator() == IBinaryOpInstruction.Operator.SUB) return left.minus(right);
        if (binOp.getOperator() == IBinaryOpInstruction.Operator.MUL) {
          if (right.isConstant()) return left.scale(right.constant());
          if (left.isConstant()) return right.scale(left.constant());
        }
        return atom(node, vn);
      }
      Integer value =
          resolveIntFlowSensitively(
              builder, node, vn, new HashSet<>(), FLOW_SENSITIVE_CONSTANT_DEPTH_CAP);
      return value != null ? constant(value) : atom(node, vn);
    }

    @Override
    public String toString() {
      return this.coefficients + " + " + this.constant;
    }
  }
}
