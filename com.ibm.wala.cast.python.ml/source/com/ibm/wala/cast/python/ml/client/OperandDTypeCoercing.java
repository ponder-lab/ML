package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.cast.python.ml.types.TensorFlowTypes.DType;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PropagationCallGraphBuilder;
import java.util.Collections;
import java.util.Map;
import java.util.Set;

/**
 * A generator whose operation coerces an operand's dtype as eager execution evaluates it (<a
 * href="https://github.com/wala/ML/issues/828">wala/ML#828</a>). TensorFlow converts a NumPy
 * operand through the dtype of the operand beside it, so a value's dtype where it is <em>fed</em>
 * need not be its dtype where it is <em>used</em>, and only the latter survives tracing.
 *
 * <p>The engine collects these coercions per operand destination and, for a parameter, rewrites the
 * incoming dtype with the one collected. Two properties make that collection sound without any
 * traversal, both checked on TF 2.9.3 in the closing analysis of <a
 * href="https://github.com/ponder-lab/Hybridize-Functions-Refactoring/issues/861">Hybridize#861</a>:
 *
 * <ul>
 *   <li><b>Chains cannot widen the set.</b> Once the argument has been coerced at its first
 *       consumer it is a tensor, and a tensor-tensor dtype mismatch raises eagerly, so a program
 *       that runs eagerly is already dtype-consistent downstream. The permissiveness applies to a
 *       NumPy operand and never to a tensor one, so consumers past the first contribute nothing new
 *       and only <em>direct</em> operands are constrained here.
 *   <li><b>Parallel direct consumptions really do disagree.</b> {@code W32 * x} and {@code V64 * x}
 *       each succeed alone, so a parameter consumed by both has no single eager-effective dtype.
 *       The engine reports the union over the imposed dtypes there (wala/ML#829): the parameter
 *       genuinely is computed at more than one dtype, at different ops, and the union states
 *       exactly that.
 * </ul>
 *
 * <p>The set is collected from the generators that implement this interface, so an operation family
 * that coerces but does not yet implement it contributes no constraint rather than a wrong one; the
 * cost of the omission is a parameter left at its fed dtype, which is the behavior that predates
 * this mechanism.
 */
public interface OperandDTypeCoercing {

  /**
   * Derives, for each operand this operation coerces, the dtype eager execution imposes on it.
   *
   * @param builder The propagation call graph builder.
   * @return A map from an operand's {@link PointerKey} to the dtype imposed on it; empty when the
   *     operation imposes nothing, which includes every case where the operand beside it does not
   *     resolve to a single definite dtype.
   */
  Map<PointerKey, DType> getOperandDTypeCoercions(PropagationCallGraphBuilder builder);

  /**
   * Derives the operands this operation consumes beside a partner whose imposed dtype it could not
   * establish: the partner is itself undecided (a parameter) or does not resolve to a single
   * definite dtype. Such an operand's coercion account is incomplete rather than absent by design
   * (a scalar-literal partner imposes nothing at run time and is not reported here), so the engine
   * treats the consumer as unaccounted for that operand and declines it to the fed dtype
   * (wala/ML#829) — without this, a declared-but-unresolved consumer would vanish from the account
   * and the remaining consumers would pin or union with false confidence.
   *
   * @param builder The propagation call graph builder.
   * @return The pointer keys of operands whose partner's imposed dtype could not be established.
   */
  default Set<PointerKey> getUnresolvedCoercionOperands(PropagationCallGraphBuilder builder) {
    return Collections.emptySet();
  }
}
