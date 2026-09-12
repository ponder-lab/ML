package com.ibm.wala.cast.python.ipa.summaries;

import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.ipa.callgraph.PythonSSAPropagationCallGraphBuilder;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.ssa.ConstantValue;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.Selector;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.Map;

public class PythonComprehensionTrampolines implements MethodTargetSelector {
  private final MethodTargetSelector base;
  private final Map<IClass, PythonSummarizedFunction> trampolines = HashMapFactory.make();

  public PythonComprehensionTrampolines(MethodTargetSelector base) {
    this.base = base;
  }

  @Override
  public IMethod getCalleeTarget(CGNode caller, CallSiteReference site, IClass receiver) {
    MethodReference method = site.getDeclaredTarget();
    if (method.getSelector().equals(AstMethodReference.fnSelector)
        && caller
            .getClassHierarchy()
            .isSubclassOf(
                receiver, caller.getClassHierarchy().lookupClass(PythonTypes.comprehension))
        && !trampolines.values().contains(caller.getMethod())) {

      if (trampolines.containsKey(receiver)) {
        return trampolines.get(receiver);

      } else {

        MethodReference synth =
            MethodReference.findOrCreate(
                method.getDeclaringClass(),
                new Selector(
                    Atom.findOrCreateUnicodeAtom("__" + receiver.getName()),
                    method.getSelector().descriptor()));

        // Only Python IR allocates a comprehension's code type, so the call reaching one is a
        // Python invoke.
        PythonInvokeInstruction inst = (PythonInvokeInstruction) caller.getIR().getCalls(site)[0];
        // The iterables are the invoke's positional parameters from the third onward, so the
        // per-element argument list and the iterable loop count positional parameters, never all
        // uses: the comprehension's `if` filters ride as keyword parameters and must not be read
        // as iterables (wala/ML#917). The summary's own parameter count and the value-number base
        // still cover every use, since a keyword parameter is still a parameter of the trampoline.
        int uses = inst.getNumberOfUses();
        int positional = inst.getNumberOfPositionalParameters();
        int v = uses + 3;
        int[] args = new int[positional - 1];
        args[0] = 1;
        int nullVal = v++;

        PythonSummary x = new PythonSummary(synth, uses);
        int idx = 0;

        // The builder binds a keyword argument to the callee's parameter of the same local name,
        // so each keyword slot after the positionals is named after its keyword; without the name
        // the filter values would never arrive (wala/ML#917).
        Map<Integer, Atom> names = HashMapFactory.make();
        int keywordSlot = positional + 1;
        for (String keyword : inst.getKeywords()) {
          names.put(keywordSlot++, Atom.findOrCreateUnicodeAtom(keyword));
        }
        x.setValueNames(names);

        x.addConstant(nullVal, null);

        int ofv = -1;
        for (int lst = 3; lst <= positional; lst++) {
          int fv = v++;
          ofv = fv;
          int lv = v++;
          x.addStatement(
              PythonLanguage.Python.instructionFactory()
                  .EachElementGetInstruction(idx++, fv, lst, nullVal));
          x.addStatement(
              PythonLanguage.Python.instructionFactory().PropertyRead(idx++, lv, lst, fv));
          args[lst - 2] = lv;
        }

        @SuppressWarnings({"unchecked", "rawtypes"})
        Pair<String, Integer>[] keywordParams = new Pair[0];

        // Each filter function (a keyword parameter) is invoked on the same per-element arguments
        // as the element lambda, so the filter bodies enter the call graph. The result is not
        // consulted: the element is stored whether or not the filter would admit it, the
        // over-approximation a test the analysis cannot decide requires (wala/ML#917).
        for (int slot = positional + 1; slot <= uses; slot++) {
          int[] filterArgs = args.clone();
          filterArgs[0] = slot;
          int fs = idx++;
          x.addStatement(
              new PythonInvokeInstruction(
                  fs,
                  v++,
                  v++,
                  new DynamicCallSiteReference(PythonTypes.CodeBody, fs),
                  filterArgs,
                  keywordParams));
        }

        int s = idx++;
        int r = v++;
        CallSiteReference ss = new DynamicCallSiteReference(PythonTypes.CodeBody, s);
        x.addStatement(new PythonInvokeInstruction(s, r, v++, ss, args, keywordParams));

        x.addStatement(PythonLanguage.Python.instructionFactory().PropertyWrite(idx++, 2, ofv, r));

        // The reflected write above keys the element by the input iterable's field, which an
        // opaque input cannot enumerate; also store the element under the append-contents
        // property, the same durable channel `xs.append(v)` uses, so container-element consumers
        // (value iteration, subscript reads, element feeds) observe comprehension-built elements
        // exactly like append-built ones (wala/ML#773).
        int contentsKey = v++;
        x.addConstant(
            contentsKey,
            new ConstantValue(PythonSSAPropagationCallGraphBuilder.LIST_APPEND_CONTENTS_FIELD));
        x.addStatement(
            PythonLanguage.Python.instructionFactory().PropertyWrite(idx++, 2, contentsKey, r));

        x.addStatement(
            PythonLanguage.Python.instructionFactory().ReturnInstruction(idx++, 2, false));

        PythonSummarizedFunction code = new PythonSummarizedFunction(synth, x, receiver);

        trampolines.put(receiver, code);

        return code;
      }
    }

    return base.getCalleeTarget(caller, site, receiver);
  }
}
