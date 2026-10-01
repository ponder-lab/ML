package com.ibm.wala.cast.python.ipa.callgraph;

import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummary;
import com.ibm.wala.cast.python.loader.PythonLoader.PythonClass;
import com.ibm.wala.cast.python.loader.PythonLoader.PythonCodeBody;
import com.ibm.wala.cast.python.loader.PythonLoader.PythonSummaryShellClass;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.Map;
import java.util.logging.Logger;

/**
 * Dispatches the application of a bare decorator ({@code @d}) by who defines the decorator
 * (wala/ML#188). A decorator defined in the analyzed program is applied the way Python applies it,
 * {@code d(f)}. A decorator modeled by a library summary is a factory: the summaries are written
 * for {@code d()(f)}, the configuring call followed by the application (see, e.g., {@code
 * tf.function} and the {@code pytest.mark} decorators), because the front end once applied every
 * decorator that way. For those, the site dispatches to an adapter that performs the two calls, so
 * a summary serves its bare and its parenthesized form alike.
 */
public class BareDecoratorTargetSelector implements MethodTargetSelector {

  private static final Logger LOGGER =
      Logger.getLogger(BareDecoratorTargetSelector.class.getName());

  /** The name of the adapter method synthesized on a summary decorator's class. */
  private static final String ADAPTER_METHOD_NAME = "bare_decorator_factory";

  private final MethodTargetSelector base;

  /** The adapters already synthesized, one per summary decorator class. */
  private final Map<IClass, IMethod> adapters = HashMapFactory.make();

  /**
   * Creates the selector.
   *
   * @param base The selector consulted for every site but a bare decorator's application to a
   *     summary-modeled decorator.
   */
  public BareDecoratorTargetSelector(MethodTargetSelector base) {
    this.base = base;
  }

  @Override
  public IMethod getCalleeTarget(CGNode caller, CallSiteReference site, IClass receiver) {
    if (site instanceof BareDecoratorCallSiteReference
        && receiver != null
        && !isProgramDefined(receiver)) {
      LOGGER.fine(() -> "Applying the summary decorator " + receiver + " as a factory.");
      return adapters.computeIfAbsent(receiver, BareDecoratorTargetSelector::makeAdapter);
    }
    return base.getCalleeTarget(caller, site, receiver);
  }

  /**
   * Whether the given class is defined by the analyzed program: a function (its code body) or a
   * source class, whose instances and constructors dispatch through their own trampolines. A
   * summary class shell is a {@link PythonClass} too and is excluded.
   *
   * @param receiver The decorator's class.
   * @return {@code true} iff the decorator is defined in the analyzed program.
   */
  private static boolean isProgramDefined(IClass receiver) {
    return receiver instanceof PythonCodeBody
        || (receiver instanceof PythonClass && !(receiver instanceof PythonSummaryShellClass));
  }

  /**
   * Synthesizes the adapter for a summary decorator's class: given the decorator {@code d} and the
   * function {@code f}, it returns {@code d()(f)}.
   *
   * @param receiver The decorator's class, hosting the adapter.
   * @return The adapter method.
   */
  private static IMethod makeAdapter(IClass receiver) {
    MethodReference ref =
        MethodReference.findOrCreate(
            receiver.getReference(),
            Atom.findOrCreateUnicodeAtom(ADAPTER_METHOD_NAME),
            AstMethodReference.fnDesc);

    // Value numbers 1 and 2 are the decorator and the decorated function.
    int decorator = 1;
    int function = 2;
    int factory = 3;
    int factoryException = 4;
    int result = 5;
    int resultException = 6;

    PythonSummary x = new PythonSummary(ref, 2);
    @SuppressWarnings({"unchecked", "rawtypes"})
    Pair<String, Integer>[] noKeywords = new Pair[0];

    int pc = 0;
    x.addStatement(
        new PythonInvokeInstruction(
            pc,
            factory,
            factoryException,
            new DynamicCallSiteReference(PythonTypes.CodeBody, pc),
            new int[] {decorator},
            noKeywords));
    pc++;

    x.addStatement(
        new PythonInvokeInstruction(
            pc,
            result,
            resultException,
            new DynamicCallSiteReference(PythonTypes.CodeBody, pc),
            new int[] {factory, function},
            noKeywords));
    pc++;

    x.addStatement(new SSAReturnInstruction(pc, result, false));

    return new PythonSummarizedFunction(ref, x, receiver);
  }
}
