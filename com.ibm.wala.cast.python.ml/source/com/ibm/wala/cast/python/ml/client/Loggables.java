package com.ibm.wala.cast.python.ml.client;

import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceFieldKey;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.LocalPointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointsToSetVariable;
import com.ibm.wala.util.intset.OrdinalSet;

/**
 * Context-free renderers for logging pointer-analysis and call-graph values. Their own {@code
 * toString()} methods render the enclosing {@link com.ibm.wala.ipa.callgraph.Context}, whose
 * scope-mapping and receiver contexts can reference each other cyclically and recurse until the
 * heap is exhausted on large graphs (e.g., nlpgnn). These helpers render only bounded, context-free
 * fields (value numbers, method signatures, allocation sites), so a {@code LOGGER.fine("..." +
 * describe(x))} is safe at any logging level. See <a
 * href="https://github.com/wala/ML/issues/697">wala/ML#697</a>.
 */
final class Loggables {

  /** The most instance keys {@link #describe(OrdinalSet)} renders before truncating. */
  private static final int MAX_SET_ELEMENTS = 10;

  private Loggables() {}

  /**
   * Describes a points-to variable by its pointer key.
   *
   * @param variable The variable to describe, or {@code null}.
   * @return A bounded, context-free description.
   */
  static String describe(PointsToSetVariable variable) {
    return variable == null ? "null" : describe(variable.getPointerKey());
  }

  /**
   * Describes a pointer key using only context-free fields.
   *
   * @param pk The pointer key to describe, or {@code null}.
   * @return A bounded, context-free description.
   */
  static String describe(PointerKey pk) {
    if (pk == null) return "null";
    if (pk instanceof LocalPointerKey) {
      LocalPointerKey lpk = (LocalPointerKey) pk;
      return "v" + lpk.getValueNumber() + " in " + signature(lpk.getNode());
    }
    if (pk instanceof InstanceFieldKey) {
      InstanceFieldKey ifk = (InstanceFieldKey) pk;
      return ifk.getField().getName() + " on " + describe(ifk.getInstanceKey());
    }
    return pk.getClass().getSimpleName();
  }

  /**
   * Describes a call-graph node by its method signature, omitting its context.
   *
   * @param node The node to describe, or {@code null}.
   * @return A bounded, context-free description.
   */
  static String describe(CGNode node) {
    return signature(node);
  }

  /**
   * Describes an instance key by its allocation site, omitting the allocating node's context.
   *
   * @param ik The instance key to describe, or {@code null}.
   * @return A bounded, context-free description.
   */
  static String describe(InstanceKey ik) {
    if (ik == null) return "null";
    if (ik instanceof AllocationSiteInNode) {
      AllocationSiteInNode asin = (AllocationSiteInNode) ik;
      return asin.getSite() + " in " + signature(asin.getNode());
    }
    return ik.getClass().getSimpleName();
  }

  /**
   * Describes a set of instance keys element-wise, rendering at most {@link #MAX_SET_ELEMENTS} of
   * them so a large points-to set cannot materialize a large string.
   *
   * @param set The set to describe, or {@code null}.
   * @return A bounded, context-free description.
   */
  static String describe(OrdinalSet<InstanceKey> set) {
    if (set == null) return "null";
    StringBuilder sb = new StringBuilder("[");
    int rendered = 0;
    for (InstanceKey ik : set) {
      if (rendered == MAX_SET_ELEMENTS) {
        sb.append(", ... (").append(set.size()).append(" total)");
        break;
      }
      if (rendered > 0) sb.append(", ");
      sb.append(describe(ik));
      rendered++;
    }
    return sb.append("]").toString();
  }

  private static String signature(CGNode node) {
    // The context tag is the context's identity hash: bounded and recursion-free (unlike the
    // context's own toString, wala/ML#697), and stable within a run, so log lines for the same
    // node in different receiver-keyed contexts are distinguishable (wala/ML#739).
    return node == null
        ? "null"
        : node.getMethod().getSignature()
            + " [ctx#"
            + Integer.toHexString(System.identityHashCode(node.getContext()))
            + "]";
  }
}
