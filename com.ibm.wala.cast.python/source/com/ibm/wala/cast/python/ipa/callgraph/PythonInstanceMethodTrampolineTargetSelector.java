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
package com.ibm.wala.cast.python.ipa.callgraph;

import static com.ibm.wala.cast.python.types.PythonTypes.CALLABLE_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.CALLABLE_METHOD_NAME_FOR_KERAS_MODELS;
import static com.ibm.wala.cast.python.types.PythonTypes.DO_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.KERAS_BUILD_METHOD_NAME;
import static com.ibm.wala.cast.python.types.PythonTypes.STATIC_METHOD;
import static com.ibm.wala.cast.python.types.Util.getDeclaringClassTypeReference;
import static com.ibm.wala.cast.python.util.Util.getAllocationSiteInNode;
import static com.ibm.wala.cast.python.util.Util.isClassMethod;
import static com.ibm.wala.types.annotations.Annotation.make;

import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.client.PythonAnalysisEngine;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummary;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.IPythonClass;
import com.ibm.wala.cast.python.loader.PythonLoader.PythonSummaryShellClass;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.classLoader.NewSiteReference;
import com.ibm.wala.classLoader.SyntheticClass;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.ipa.callgraph.propagation.AllocationSiteInNode;
import com.ibm.wala.ipa.callgraph.propagation.InstanceKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKey;
import com.ibm.wala.ipa.callgraph.propagation.PointerKeyFactory;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ipa.summaries.BypassSyntheticClass;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.types.ClassLoaderReference;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.Selector;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import com.ibm.wala.util.intset.OrdinalSet;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

public class PythonInstanceMethodTrampolineTargetSelector<T>
    extends PythonMethodTrampolineTargetSelector<T> {

  private static final Logger LOGGER =
      Logger.getLogger(PythonInstanceMethodTrampolineTargetSelector.class.getName());

  private PythonAnalysisEngine<T> engine;

  public PythonInstanceMethodTrampolineTargetSelector(
      MethodTargetSelector base, PythonAnalysisEngine<T> engine) {
    super(base);
    this.engine = engine;
  }

  @Override
  protected boolean shouldProcess(CGNode caller, CallSiteReference site, IClass receiver) {
    IClassHierarchy cha = receiver.getClassHierarchy();
    return cha.isSubclassOf(receiver, cha.lookupClass(PythonTypes.trampoline))
        || this.isCallable(receiver);
  }

  @Override
  public IMethod getCalleeTarget(CGNode caller, CallSiteReference site, IClass receiver) {
    // TODO: Callable detection may need to be moved. See https://github.com/wala/ML/issues/207. If
    // it stays here, we should further document the receiver swapping process.
    if (isCallable(receiver)) {
      LOGGER.finer("Encountered callable.");

      PythonInvokeInstruction call = this.getCall(caller, site);
      if (call == null) return super.getCalleeTarget(caller, site, receiver);

      Set<IClass> callables = getCallables(caller, receiver.getClassHierarchy(), call);

      if (callables.isEmpty()) return null; // not found.

      if (callables.size() > 1) {
        // Multiple callable classes may arrive at this site. Declining to dispatch here would
        // drop targets the analysis just found, producing NO call edges where real targets exist
        // (wala/ML#869): a may-analysis must not answer a choice it cannot make with silence.
        // Dispatch instead through a fan-out trampoline that reads each arriving instance's own
        // bound-callable field, so every candidate dispatches and the per-instance bound objects
        // do the discrimination. The WARNING below is a PERMANENT DIAGNOSTIC, not a leftover: the
        // fan-out size distribution is what bounds this path's imprecision, it has been measured
        // on one subject only (maximum three, no tail), and this message plus the fan-out-named
        // trampoline in the graph are how that finding is checked in the field.
        LOGGER.warning(
            "Multiple ("
                + callables.size()
                + ") callable targets found; dispatching to all of them (wala/ML#869).");

        return getFanoutDispatcher(call, callables.size(), receiver.getClassHierarchy());
      }

      // It's a callable. Change the receiver.
      receiver = callables.iterator().next();
      LOGGER.finer("Substituting the receiver with one derived from a callable.");
    }

    return super.getCalleeTarget(caller, site, receiver);
  }

  /**
   * The fan-out dispatchers already synthesized, keyed by trampoline name, which encodes the call's
   * positional/keyword layout and the fan-out size. The body is candidate-set-independent (it reads
   * the arriving instance's own bound-callable fields), so sites whose calls share a layout share a
   * body; the fan-out size participates in the name purely as a graph-visible diagnostic.
   */
  private final Map<String, IMethod> fanoutDispatchers = HashMapFactory.make();

  /**
   * The synthetic field through which the fan-out dispatcher merges the receiver's bound callables
   * (wala/ML#869): the summary language has no phi, and the pointer analysis is flow-insensitive,
   * so writing several values to one field of a scratch object and reading the field back unions
   * them into a single value.
   */
  private static final String FANOUT_CALLABLES_FIELD = "$fanoutCallables";

  /**
   * Returns the fan-out dispatcher for a callable call site whose receiver's points-to set spans
   * several callable classes (wala/ML#869). The dispatcher receives the RAW instances as its
   * receiver parameter (the single-candidate path's receiver substitution applies only when a
   * concrete candidate class is chosen), so it reaches each instance's bound-callable object the
   * way the program would: it reads the {@code __call__}, {@code call}, and {@code do} convention
   * fields off the instance (a missing convention yields an empty points-to set and contributes
   * nothing, exactly as an unmodeled callable does today), merges them through a scratch field, and
   * forwards the call's positional and keyword arguments to the merged value. Each instance's field
   * holds its OWN class's bound object, whose per-class trampoline (with its class filter and the
   * wala/ML#595 build hook) then dispatches, so discrimination is per-instance downstream and the
   * fan-out here adds edges only for classes the points-to analysis says may arrive.
   *
   * @param call The {@link PythonInvokeInstruction} whose layout the dispatcher forwards.
   * @param fanout The number of candidate callable classes, for the diagnostic name.
   * @param cha The class hierarchy hosting the trampoline class.
   * @return The dispatcher method.
   */
  private IMethod getFanoutDispatcher(
      PythonInvokeInstruction call, int fanout, IClassHierarchy cha) {
    String name = this.getTrampolineName(call) + "$fanout" + fanout;

    if (!fanoutDispatchers.containsKey(name)) {
      MethodReference tr =
          MethodReference.findOrCreate(
              PythonTypes.trampoline,
              Atom.findOrCreateUnicodeAtom(name),
              AstMethodReference.fnDesc);
      PythonSummary x = new PythonSummary(tr, call.getNumberOfTotalParameters());
      int v = call.getNumberOfTotalParameters() + 1;

      Map<Integer, Atom> names = HashMapFactory.make();
      int vScratch = v;
      int vMerged = v + 4;
      int pc = 0;

      x.addStatement(
          PythonLanguage.Python.instructionFactory()
              .NewInstruction(pc, vScratch, NewSiteReference.make(pc, PythonTypes.object)));
      pc++;

      FieldReference mergeField =
          FieldReference.findOrCreate(
              PythonTypes.Root,
              Atom.findOrCreateUnicodeAtom(FANOUT_CALLABLES_FIELD),
              PythonTypes.Root);

      String[] conventions = {
        CALLABLE_METHOD_NAME, CALLABLE_METHOD_NAME_FOR_KERAS_MODELS, DO_METHOD_NAME
      };

      for (int c = 0; c < conventions.length; c++) {
        int vBound = v + 1 + c;

        x.addStatement(
            PythonLanguage.Python.instructionFactory()
                .GetInstruction(
                    pc++,
                    vBound,
                    1,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom(conventions[c]),
                        PythonTypes.Root)));

        x.addStatement(
            PythonLanguage.Python.instructionFactory()
                .PutInstruction(pc++, vScratch, vBound, mergeField));
      }

      x.addStatement(
          PythonLanguage.Python.instructionFactory()
              .GetInstruction(pc++, vMerged, vScratch, mergeField));

      int i = 0;
      int[] params = new int[Math.max(1, call.getNumberOfPositionalParameters())];
      params[i++] = vMerged;

      for (int j = 1; j < call.getNumberOfPositionalParameters(); j++) params[i++] = j + 1;

      int ki = 0, ji = call.getNumberOfPositionalParameters() + 1;
      @SuppressWarnings({"unchecked", "rawtypes"})
      Pair<String, Integer>[] keys = new Pair[0];

      if (call.getKeywords() != null) {
        @SuppressWarnings({"unchecked", "rawtypes"})
        Pair<String, Integer>[] tmp = (Pair<String, Integer>[]) new Pair[call.getKeywords().size()];
        keys = tmp;

        for (String k : call.getKeywords()) {
          names.put(ji, Atom.findOrCreateUnicodeAtom(k));
          keys[ki++] = Pair.<String, Integer>make(k, ji++);
        }
      }

      int result = v + 5;
      int except = v + 6;

      x.addStatement(
          new PythonInvokeInstruction(
              pc,
              result,
              except,
              new DynamicCallSiteReference(call.getCallSite().getDeclaredTarget(), pc),
              params,
              keys));
      pc++;

      x.addStatement(new SSAReturnInstruction(pc, result, false));
      x.setValueNames(names);

      fanoutDispatchers.put(
          name, new PythonSummarizedFunction(tr, x, cha.lookupClass(PythonTypes.trampoline)));
    }

    return fanoutDispatchers.get(name);
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  @Override
  protected void populate(
      PythonSummary x, int v, IClass receiver, PythonInvokeInstruction call, Logger logger) {
    Map<Integer, Atom> names = HashMapFactory.make();
    IClass filter = ((PythonInstanceMethodTrampoline) receiver).getRealClass();

    x.addStatement(
        PythonLanguage.Python.instructionFactory()
            .GetInstruction(
                0,
                v,
                1,
                FieldReference.findOrCreate(
                    PythonTypes.Root,
                    Atom.findOrCreateUnicodeAtom("$function"),
                    PythonTypes.Root)));

    int v0 = v + 1;

    x.addStatement(
        PythonLanguage.Python.instructionFactory()
            .CheckCastInstruction(1, v0, v, filter.getReference(), true));

    int v1;

    // Are we calling a static method?
    boolean staticMethodReceiver = filter.getAnnotations().contains(make(STATIC_METHOD));
    logger.fine(
        staticMethodReceiver
            ? "Found static method receiver: " + filter
            : "Method is not static: " + filter);

    // Are we calling a class method? If so, it would be using an object instance instead of a
    // class on the LHS.
    boolean classMethodReceiver = isClassMethod(receiver);

    // only add self if the receiver isn't static or a class method.
    if (!staticMethodReceiver && !classMethodReceiver) {
      v1 = v + 2;

      x.addStatement(
          PythonLanguage.Python.instructionFactory()
              .GetInstruction(
                  1,
                  v1,
                  1,
                  FieldReference.findOrCreate(
                      PythonTypes.Root, Atom.findOrCreateUnicodeAtom("$self"), PythonTypes.Root)));

      // The Keras layer-call protocol builds lazily: `Layer.__call__` invokes `self.build(...)`
      // before the first `call`, and user subclasses commonly create their sublayers there (e.g.
      // `self._kernel = tf.keras.layers.Dense(...)`). Nothing else in the analysis invokes
      // `build`, so without this the sublayers those bodies create have empty points-to sets and
      // every value flowing through them unravels (wala/ML#595). Emitting the invocation
      // unconditionally is safe: a class without a `build` method yields an empty points-to set
      // for the field read, and the invoke has no targets.
      if (filter
          .getReference()
          .getName()
          .toString()
          .endsWith("/" + CALLABLE_METHOD_NAME_FOR_KERAS_MODELS)) {
        int buildFunction = v1 + 3;
        x.addStatement(
            PythonLanguage.Python.instructionFactory()
                .GetInstruction(
                    2,
                    buildFunction,
                    v1,
                    FieldReference.findOrCreate(
                        PythonTypes.Root,
                        Atom.findOrCreateUnicodeAtom(KERAS_BUILD_METHOD_NAME),
                        PythonTypes.Root)));
        x.addStatement(
            new PythonInvokeInstruction(
                3,
                v1 + 4,
                v1 + 5,
                new DynamicCallSiteReference(call.getCallSite().getDeclaredTarget(), 3),
                new int[] {buildFunction, v1},
                new Pair[0]));
      }
    } else if (classMethodReceiver) {
      // Add a class reference.
      v1 = v + 2;

      x.addStatement(
          PythonLanguage.Python.instructionFactory()
              .GetInstruction(
                  1,
                  v1,
                  1,
                  FieldReference.findOrCreate(
                      PythonTypes.Root, Atom.findOrCreateUnicodeAtom("$class"), PythonTypes.Root)));

      int v2 = v + 3;
      TypeReference reference = getDeclaringClassTypeReference(filter.getReference());

      x.addStatement(
          PythonLanguage.Python.instructionFactory()
              .CheckCastInstruction(1, v2, v1++, reference, true));
    } else v1 = v + 1;

    int i = 0;
    int paramSize =
        Math.max(
            staticMethodReceiver ? 1 : 2,
            call.getNumberOfPositionalParameters() + (staticMethodReceiver ? 0 : 1));
    int[] params = new int[paramSize];
    params[i++] = v0;

    if (!staticMethodReceiver) params[i++] = v1;

    for (int j = 1; j < call.getNumberOfPositionalParameters(); j++) params[i++] = j + 1;

    int ki = 0, ji = call.getNumberOfPositionalParameters() + 1;
    @SuppressWarnings({"unchecked", "rawtypes"})
    Pair<String, Integer>[] keys = new Pair[0];

    if (call.getKeywords() != null) {
      @SuppressWarnings({"unchecked", "rawtypes"})
      Pair<String, Integer>[] tmp = (Pair<String, Integer>[]) new Pair[call.getKeywords().size()];
      keys = tmp;

      for (String k : call.getKeywords()) {
        names.put(ji, Atom.findOrCreateUnicodeAtom(k));
        keys[ki++] = Pair.<String, Integer>make(k, ji++);
      }
    }

    int result = v1 + 1;
    int except = v1 + 2;

    CallSiteReference ref = new DynamicCallSiteReference(call.getCallSite().getDeclaredTarget(), 2);

    x.addStatement(new PythonInvokeInstruction(2, result, except, ref, params, keys));
    x.addStatement(new SSAReturnInstruction(3, result, false));
    x.setValueNames(names);
  }

  /**
   * Returns the callable classes of the receiver of the given {@link PythonInvokeInstruction}: one
   * per callable class over the receiver's whole points-to set, keyed by each member's allocating
   * node's declaring class. The caller decides what to do with the set's size; this method no
   * longer collapses multiple candidates to nothing (wala/ML#869).
   *
   * @param caller The {@link CGNode} representing the caller of the given {@link
   *     PythonInvokeInstruction}.
   * @param cha The receiver's {@link IClassHierarchy}.
   * @param call The {@link PythonInvokeInstruction} in question.
   * @return The callable classes of the given {@link PythonInvokeInstruction}'s receiver, empty
   *     when none resolve.
   */
  private Set<IClass> getCallables(
      CGNode caller, IClassHierarchy cha, PythonInvokeInstruction call) {
    PythonSSAPropagationCallGraphBuilder builder = this.getEngine().getCachedCallGraphBuilder();

    // Lookup the callable method.
    PointerKeyFactory pkf = builder.getPointerKeyFactory();
    PointerKey receiver = pkf.getPointerKeyForLocal(caller, call.getUse(0));
    OrdinalSet<InstanceKey> objs = builder.getPointerAnalysis().getPointsToSet(receiver);

    // The set of potential callables to be returned.
    Set<IClass> callableSet = new HashSet<>();

    for (InstanceKey o : objs) {
      AllocationSiteInNode instanceKey = getAllocationSiteInNode(o);
      if (instanceKey != null) {
        CGNode node = instanceKey.getNode();
        IMethod method = node.getMethod();
        IClass declaringClass = method.getDeclaringClass();
        final ClassLoaderReference classLoaderReference =
            declaringClass.getClassLoader().getReference();

        // First, check the concrete type of the allocated object
        IClass concreteType = o.concreteType();
        if (concreteType != null) {
          String concreteTypeName = "$" + concreteType.getName().toString().substring(1);
          IClass concreteCallable =
              cha.lookupClass(
                  TypeReference.findOrCreateClass(
                      classLoaderReference, concreteTypeName, CALLABLE_METHOD_NAME));
          if (concreteCallable == null) {
            concreteCallable =
                cha.lookupClass(
                    TypeReference.findOrCreateClass(
                        classLoaderReference,
                        concreteTypeName,
                        CALLABLE_METHOD_NAME_FOR_KERAS_MODELS));
          }
          if (concreteCallable == null) {
            concreteCallable =
                cha.lookupClass(
                    TypeReference.findOrCreateClass(
                        classLoaderReference, concreteTypeName, DO_METHOD_NAME));
          }
          if (concreteCallable != null) {
            callableSet.add(concreteCallable);
            continue;
          }
        }

        TypeName declaringClassName = declaringClass.getName();
        final String packageName = "$" + declaringClassName.toString().substring(1);

        IClass callable =
            cha.lookupClass(
                TypeReference.findOrCreateClass(
                    classLoaderReference, packageName, CALLABLE_METHOD_NAME));

        if (callable == null) {
          callable =
              cha.lookupClass(
                  TypeReference.findOrCreateClass(
                      classLoaderReference,
                      declaringClassName.toString().substring(1),
                      CALLABLE_METHOD_NAME));
        }

        if (callable == null) {
          callable =
              cha.lookupClass(
                  TypeReference.findOrCreateClass(
                      classLoaderReference, packageName, DO_METHOD_NAME));

          if (callable == null) {
            callable =
                cha.lookupClass(
                    TypeReference.findOrCreateClass(
                        classLoaderReference,
                        declaringClassName.toString().substring(1),
                        DO_METHOD_NAME));
          }
        }

        // The Keras `call` convention (https://github.com/wala/ML/issues/106) applies to ANY
        // class with a `call` method, without checking its hierarchy. Although a subclass's
        // summary-modeled base has been resolvable since the class shells of
        // https://github.com/wala/ML/issues/118, gating this on a shell ancestor drops sound
        // dispatch for every subclass whose base does NOT resolve (cross-module imports,
        // https://github.com/wala/ML/issues/571; bare-name collisions,
        // https://github.com/wala/ML/issues/657; unmodeled spellings) and empirically loses 18
        // tests' worth of forward-pass coverage. Tightening is tracked by
        // https://github.com/wala/ML/issues/663.
        if (callable == null) {
          LOGGER.finer(
              "Attempting the Keras `call` convention for"
                  + " https://github.com/wala/ML/issues/106.");

          callable =
              cha.lookupClass(
                  TypeReference.findOrCreateClass(
                      classLoaderReference, packageName, CALLABLE_METHOD_NAME_FOR_KERAS_MODELS));

          if (callable == null) {
            callable =
                cha.lookupClass(
                    TypeReference.findOrCreateClass(
                        classLoaderReference,
                        declaringClassName.toString().substring(1),
                        CALLABLE_METHOD_NAME_FOR_KERAS_MODELS));
          }

          if (callable != null)
            LOGGER.info(
                "Applying the Keras `call` convention for"
                    + " https://github.com/wala/ML/issues/106.");
        }

        if (callable != null) {
          callableSet.add(callable);
        }
      }
    }

    return callableSet;
  }

  public PythonAnalysisEngine<T> getEngine() {
    return engine;
  }

  @Override
  protected Logger getLogger() {
    return LOGGER;
  }

  /**
   * Returns true iff the given {@link IClass} represents a Python callable object.
   *
   * @param receiver The {@link IClass} in question.
   * @return True iff the given {@link IClass} represents a Python callable object.
   */
  private boolean isCallable(IClass receiver) {
    if (receiver == null) return false;
    if (receiver.getReference().equals(PythonTypes.object)
        || receiver instanceof BypassSyntheticClass) {
      return true;
    }
    if (receiver instanceof SyntheticClass
        && (receiver.getMethod(
                    new Selector(
                        Atom.findOrCreateUnicodeAtom(CALLABLE_METHOD_NAME),
                        AstMethodReference.fnDesc))
                != null
            || receiver.getMethod(
                    new Selector(
                        Atom.findOrCreateUnicodeAtom(CALLABLE_METHOD_NAME_FOR_KERAS_MODELS),
                        AstMethodReference.fnDesc))
                != null
            || receiver.getMethod(
                    new Selector(
                        Atom.findOrCreateUnicodeAtom(DO_METHOD_NAME), AstMethodReference.fnDesc))
                != null)) {
      return true;
    }

    // A summary-modeled class whose method references include a `__call__`/`call` function class
    // is callable, whether materialized as an engine-registered synthetic class (for allocatable
    // summary types) or as a summary class shell (`PythonLoader.defineSummaryClassShell`,
    // wala/ML#106). Source classes are deliberately excluded: their instances dispatch through
    // constructor-wired trampolines, and substituting the receiver here instead drops those calls.
    if ((receiver instanceof SyntheticClass || receiver instanceof PythonSummaryShellClass)
        && receiver instanceof IPythonClass) {
      for (MethodReference mr : ((IPythonClass) receiver).getMethodReferences()) {
        String clsName = mr.getDeclaringClass().getName().toString();
        if (clsName.endsWith("/" + CALLABLE_METHOD_NAME)
            || clsName.endsWith("/" + CALLABLE_METHOD_NAME_FOR_KERAS_MODELS)) {
          return true;
        }
      }
    }
    return false;
  }
}
