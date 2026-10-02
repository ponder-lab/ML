package com.ibm.wala.cast.python.ipa.summaries;

import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.SyntheticMethod;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ipa.summaries.BypassMethodTargetSelector;
import com.ibm.wala.ipa.summaries.MethodSummary;
import com.ibm.wala.ipa.summaries.SummarizedMethod;
import com.ibm.wala.ipa.summaries.SummarizedMethodWithNames;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.util.collections.HashMapFactory;
import java.util.Map;
import java.util.Set;

/**
 * A {@link BypassMethodTargetSelector} whose synthetic methods keep their summaries' parameter
 * names (wala/ML#996).
 *
 * <p>The base selector keeps a summary's {@code paramNames} only when the class hierarchy resolves
 * no method for the call, and discards them when it does. A summary method transformed into a
 * function class (every {@code <method>} of a class other than {@code do}) is resolved by the
 * hierarchy through the class registered for it, so its names were discarded, and a keyword
 * argument at a call to it, which the builder binds to a formal by name, bound nothing. The
 * allocatable classes' {@code do} summaries took the other path and kept theirs, so {@code
 * tf.Variable(..., constraint=c)} bound {@code constraint} where {@code self.add_weight(...,
 * constraint=c)} did not.
 */
public class PythonBypassMethodTargetSelector extends BypassMethodTargetSelector {

  /** The summaries, by the method they summarize. */
  private final Map<MethodReference, MethodSummary> summaries;

  /** The synthetic methods created here, by reference; {@code null} marks a method without one. */
  private final Map<MethodReference, SummarizedMethod> named = HashMapFactory.make();

  /**
   * Constructs the selector.
   *
   * @param parent The selector consulted for methods without a summary.
   * @param summaries The summaries, by the method they summarize.
   * @param ignoredPackages The packages whose methods are summarized as no-ops.
   * @param cha The class hierarchy.
   */
  public PythonBypassMethodTargetSelector(
      MethodTargetSelector parent,
      Map<MethodReference, MethodSummary> summaries,
      Set<Atom> ignoredPackages,
      IClassHierarchy cha) {
    super(parent, summaries, ignoredPackages, cha);
    this.summaries = summaries;
  }

  /**
   * As the base selector, but the synthetic method is a {@link SummarizedMethodWithNames}, so a
   * summary's parameter names reach its IR whichever way the call resolved.
   */
  @Override
  protected SyntheticMethod findOrCreateSyntheticMethod(IMethod m, boolean isStatic) {
    MethodReference ref = m.getReference();
    if (this.named.containsKey(ref)) return this.named.get(ref);
    MethodSummary summary =
        this.canIgnore(ref) ? this.generateNoOp(ref, isStatic) : this.summaries.get(ref);
    SummarizedMethod method =
        summary == null ? null : new SummarizedMethodWithNames(ref, summary, m.getDeclaringClass());
    this.named.put(ref, method);
    return method;
  }
}
