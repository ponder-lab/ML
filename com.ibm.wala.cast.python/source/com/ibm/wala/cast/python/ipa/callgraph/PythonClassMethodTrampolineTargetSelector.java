/******************************************************************************
 * Copyright (c) 2018 IBM Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 *****************************************************************************/
package com.ibm.wala.cast.python.ipa.callgraph;

import static com.ibm.wala.cast.python.types.PythonTypes.STATIC_METHOD;
import static com.ibm.wala.cast.python.types.Util.getDeclaringClassTypeReference;
import static com.ibm.wala.cast.python.types.Util.getGlobalName;
import static com.ibm.wala.cast.python.types.Util.makeGlobalRef;
import static com.ibm.wala.cast.python.util.Util.isClassMethod;
import static com.ibm.wala.types.annotations.Annotation.make;

import com.ibm.wala.cast.ir.ssa.AstGlobalRead;
import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.ipa.summaries.PythonInstanceMethodTrampoline;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummarizedFunction;
import com.ibm.wala.cast.python.ipa.summaries.PythonSummary;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.python.types.Util;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.CallSiteReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.CGNode;
import com.ibm.wala.ipa.callgraph.MethodTargetSelector;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.SSAReturnInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.Map;
import java.util.logging.Logger;

public class PythonClassMethodTrampolineTargetSelector<T> implements MethodTargetSelector {

  private static final Logger LOGGER =
      Logger.getLogger(PythonClassMethodTrampolineTargetSelector.class.getName());

  private final MethodTargetSelector base;

  public PythonClassMethodTrampolineTargetSelector(MethodTargetSelector base) {
    this.base = base;
  }

  private final Map<Pair<IClass, Integer>, IMethod> codeBodies = HashMapFactory.make();

  @SuppressWarnings("unchecked")
  @Override
  public IMethod getCalleeTarget(CGNode caller, CallSiteReference site, IClass receiver) {
    if (receiver != null) {
      LOGGER.fine("Getting callee target for receiver: " + receiver);
      LOGGER.fine("Calling method name is: " + caller.getMethod().getName());

      IClassHierarchy cha = receiver.getClassHierarchy();

      // Are we calling a class method?
      boolean classMethodReceiver = isClassMethod(receiver);

      // Only for class methods with a class on the LHS (not an object instance).
      if (classMethodReceiver && !cha.isSubclassOf(receiver, cha.lookupClass(PythonTypes.trampoline))) {
        PythonInvokeInstruction call = (PythonInvokeInstruction) caller.getIR().getCalls(site)[0];
        Pair<IClass, Integer> key = Pair.make(receiver, call.getNumberOfTotalParameters());

        if (!codeBodies.containsKey(key)) {
          Map<Integer, Atom> names = HashMapFactory.make();
          
          MethodReference tr =
              MethodReference.findOrCreate(
                  receiver.getReference(),
                  Atom.findOrCreateUnicodeAtom("trampoline" + call.getNumberOfTotalParameters()),
                  AstMethodReference.fnDesc);
          
          PythonSummary x = new PythonSummary(tr, call.getNumberOfTotalParameters());

          int v = call.getNumberOfTotalParameters() + 1;

//          x.addStatement(
//              PythonLanguage.Python.instructionFactory()
//                  .GetInstruction(
//                      0,
//                      v,
//                      1,
//                      FieldReference.findOrCreate(
//                          PythonTypes.Root,
//                          Atom.findOrCreateUnicodeAtom("$function"),
//                          PythonTypes.Root)));
//
          int v0 = v + 1;
//
//          x.addStatement(
//              PythonLanguage.Python.instructionFactory()
//                  .CheckCastInstruction(1, v0, v, receiver.getReference(), true));
//
          int v1;
//
//          // Are we calling a static method?
//          boolean staticMethodReceiver = receiver.getAnnotations().contains(make(STATIC_METHOD));
//          LOGGER.fine(
//              staticMethodReceiver
//                  ? "Found static method receiver: " + receiver 
//                  : "Method is not static: " + receiver);
//
//          // only add self if the receiver isn't static or a class method.
//          if (!staticMethodReceiver && !classMethodReceiver) {
//            v1 = v + 2;
//
//            x.addStatement(
//                PythonLanguage.Python.instructionFactory()
//                    .GetInstruction(
//                        1,
//                        v1,
//                        1,
//                        FieldReference.findOrCreate(
//                            PythonTypes.Root,
//                            Atom.findOrCreateUnicodeAtom("$self"),
//                            PythonTypes.Root)));
//          } else if (classMethodReceiver) {
//            // Add a class reference.
//            v1 = v + 2;
//
//            x.addStatement(
//                PythonLanguage.Python.instructionFactory()
//                    .GetInstruction(
//                        1,
//                        v1,
//                        1,
//                        FieldReference.findOrCreate(
//                            PythonTypes.Root,
//                            Atom.findOrCreateUnicodeAtom("$class"),
//                            PythonTypes.Root)));
//
//            int v2 = v + 3;
//            TypeReference reference = getDeclaringClassTypeReference(receiver.getReference());
//
//            x.addStatement(
//                PythonLanguage.Python.instructionFactory()
//                    .CheckCastInstruction(1, v2, v1++, reference, true));
//          } else
          
          System.out.println(receiver);
          
          FieldReference globalRef = makeGlobalRef(receiver.getClassLoader(), "script tf2_test_class_method4.py/MyClass");
          System.out.println(globalRef);
          
          x.addStatement(new AstGlobalRead(99, 99, globalRef));
          
          v1 = v + 1;

          int i = 0;
          int paramSize =
              Math.max(
                  1,
                  call.getNumberOfPositionalParameters());
          int[] params = new int[paramSize];
          params[i++] = v0;

          for (int j = 1; j < call.getNumberOfPositionalParameters(); j++) params[i++] = j + 1;

          int ki = 0, ji = call.getNumberOfPositionalParameters() + 1;
          Pair<String, Integer>[] keys = new Pair[0];

          if (call.getKeywords() != null) {
            keys = new Pair[call.getKeywords().size()];

            for (String k : call.getKeywords()) {
              names.put(ji, Atom.findOrCreateUnicodeAtom(k));
              keys[ki++] = Pair.make(k, ji++);
            }
          }

          int result = v1 + 1;
          int except = v1 + 2;

          CallSiteReference ref =
              new DynamicCallSiteReference(call.getCallSite().getDeclaredTarget(), 2);

          x.addStatement(new PythonInvokeInstruction(2, result, except, ref, params, keys));
          x.addStatement(new SSAReturnInstruction(3, result, false));
          x.setValueNames(names);

          codeBodies.put(key, new PythonSummarizedFunction(tr, x, receiver));
        }

        return codeBodies.get(key);
      }
    }

    return base.getCalleeTarget(caller, site, receiver);
  }
}
