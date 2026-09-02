package com.ibm.wala.cast.python.ipa.callgraph;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.object;

import com.ibm.wala.cast.ir.ssa.AstInstructionFactory;
import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.PythonLoader;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IMethod;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.Entrypoint;
import com.ibm.wala.ipa.callgraph.impl.AbstractRootMethod;
import com.ibm.wala.ipa.callgraph.impl.DefaultEntrypoint;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.Pair;

/**
 * An {@link Entrypoint} of a <a href="http://pytest.org">Pytest</a> test case.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class PytesttEntrypoint extends DefaultEntrypoint {

  public PytesttEntrypoint(IMethod method, IClassHierarchy cha) {
    super(method, cha);
  }

  public PytesttEntrypoint(MethodReference method, IClassHierarchy cha) {
    super(method, cha);
  }

  /**
   * @see {@link TurtleSummary#turtleEntryPoint(IMethod)}.
   */
  @Override
  public SSAAbstractInvokeInstruction addCall(AbstractRootMethod m) {
    int paramValues[] = new int[getNumberOfParameters()];

    for (int j = 0; j < paramValues.length; j++) {
      AstInstructionFactory insts = PythonLanguage.Python.instructionFactory();

      String methodDeclaringClassName = getMethod().getDeclaringClass().getName().toString();

      if (j == 0 && methodDeclaringClassName.contains("/")) {
        int v = m.nextLocal++;
        paramValues[j] = v;

        if (getMethod().getDeclaringClass() instanceof PythonLoader.DynamicMethodBody) {
          FieldReference global =
              FieldReference.findOrCreate(
                  PythonTypes.Root,
                  Atom.findOrCreateUnicodeAtom(
                      "global "
                          + methodDeclaringClassName.substring(
                              1, methodDeclaringClassName.lastIndexOf('/'))),
                  PythonTypes.Root);

          int idx = m.statements.size();
          int cls = m.nextLocal++;
          int obj = m.nextLocal++;

          m.statements.add(insts.GlobalRead(m.statements.size(), cls, global));
          idx = m.statements.size();

          @SuppressWarnings({"unchecked", "rawtypes"})
          PythonInvokeInstruction invokeInstruction =
              new PythonInvokeInstruction(
                  idx,
                  obj,
                  m.nextLocal++,
                  new DynamicCallSiteReference(PythonTypes.CodeBody, idx),
                  new int[] {cls},
                  new Pair[0]);

          m.statements.add(invokeInstruction);
          idx = m.statements.size();

          String method = methodDeclaringClassName;
          String field = method.substring(method.lastIndexOf('/') + 1);

          FieldReference f =
              FieldReference.findOrCreate(
                  PythonTypes.Root, Atom.findOrCreateUnicodeAtom(field), PythonTypes.Root);

          m.statements.add(insts.GetInstruction(idx, v, obj, f));
        } else {
          FieldReference global =
              FieldReference.findOrCreate(
                  PythonTypes.Root,
                  Atom.findOrCreateUnicodeAtom("global " + methodDeclaringClassName.substring(1)),
                  PythonTypes.Root);

          m.statements.add(insts.GlobalRead(m.statements.size(), v, global));
        }
      } else paramValues[j] = makeArgument(m, j);

      if (paramValues[j] == -1)
        // there was a problem
        return null;

      TypeReference x[] = getParameterTypes(j);

      if (x.length == 1 && x[0].equals(object))
        m.statements.add(
            insts.PutInstruction(
                m.statements.size(),
                paramValues[j],
                paramValues[j],
                FieldReference.findOrCreate(object, Atom.findOrCreateUnicodeAtom("pytest"), Root)));
    }

    // SPIKE wala/ML#868 layer two: when the test method's decorator carries mineable class
    // arguments, bind the extra parameter to each named class's global (the same global-read idiom
    // parameter 0 uses above), one invocation per argument, mirroring the parameterized runner's
    // per-argument test cases.
    java.util.List<Integer> decoratorBindings = decoratorArgumentBindings(m);

    if (!decoratorBindings.isEmpty() && paramValues.length >= 2) {
      PythonInvokeInstruction last = null;

      for (int binding : decoratorBindings) {
        int[] boundParams = paramValues.clone();
        boundParams[1] = binding;

        int pc = m.statements.size();

        @SuppressWarnings({"unchecked", "rawtypes"})
        PythonInvokeInstruction call =
            new PythonInvokeInstruction(
                pc,
                m.nextLocal++,
                m.nextLocal++,
                new DynamicCallSiteReference(PythonTypes.CodeBody, pc),
                boundParams,
                new Pair[0]);

        m.statements.add(call);
        last = call;
      }

      return last;
    }

    int pc = m.statements.size();

    @SuppressWarnings({"unchecked", "rawtypes"})
    PythonInvokeInstruction call =
        new PythonInvokeInstruction(
            pc,
            m.nextLocal++,
            m.nextLocal++,
            new DynamicCallSiteReference(PythonTypes.CodeBody, pc),
            paramValues,
            new Pair[0]);

    m.statements.add(call);

    return call;
  }

  /**
   * SPIKE wala/ML#868: emits a global read per mineable class-name argument of the entrypoint
   * method's decorators, returning the read values. A bare name resolves in the method's own
   * script; a dotted name resolves by scanning the hierarchy for a class of that name whose
   * defining script's file name matches the qualifier.
   *
   * @param m The root method being built.
   * @return The value numbers holding each resolved class object; empty when the method carries no
   *     mineable decorator arguments.
   */
  private java.util.List<Integer> decoratorArgumentBindings(AbstractRootMethod m) {
    java.util.List<Integer> ret = new java.util.ArrayList<>();

    if (!(getMethod().getDeclaringClass() instanceof PythonLoader.DynamicMethodBody)) return ret;
    PythonLoader.DynamicMethodBody body =
        (PythonLoader.DynamicMethodBody) getMethod().getDeclaringClass();

    String methodDeclaringClassName = getMethod().getDeclaringClass().getName().toString();
    String script = methodDeclaringClassName.substring(1, methodDeclaringClassName.indexOf('/'));

    for (com.ibm.wala.cast.python.util.Util.DecoratorCall decoratorCall : body.getDecoratorCalls())
      for (String argument : decoratorCall.argumentNames()) {
        if ("None".equals(argument)
            || com.ibm.wala.cast.python.util.Util.UNMINEABLE_DECORATOR_ARGUMENT.equals(argument))
          continue;

        String globalName = resolveClassGlobal(script, argument);
        if (globalName == null) continue;

        AstInstructionFactory insts = PythonLanguage.Python.instructionFactory();
        int v = m.nextLocal++;
        m.statements.add(
            insts.GlobalRead(
                m.statements.size(),
                v,
                FieldReference.findOrCreate(
                    PythonTypes.Root,
                    Atom.findOrCreateUnicodeAtom("global " + globalName),
                    PythonTypes.Root)));
        ret.add(v);
      }

    return ret;
  }

  /**
   * SPIKE wala/ML#868: resolves a mined decorator-argument name to the defining script's global
   * name for the class object.
   *
   * @param script The entrypoint method's own script (e.g. {@code script test.py}).
   * @param argument The mined argument name, bare or dotted.
   * @return The global's name (e.g. {@code script mod.py/A}), or {@code null} when it does not
   *     resolve.
   */
  private String resolveClassGlobal(String script, String argument) {
    int dot = argument.lastIndexOf('.');

    if (dot < 0) {
      String candidate = script + "/" + argument;
      return getCha()
                  .lookupClass(
                      TypeReference.findOrCreate(PythonTypes.pythonLoader, "L" + candidate))
              != null
          ? candidate
          : null;
    }

    String qualifier = argument.substring(0, dot);
    String className = argument.substring(dot + 1);
    String wantedFile = qualifier.substring(qualifier.lastIndexOf('.') + 1) + ".py";

    for (com.ibm.wala.classLoader.IClass klass : getCha()) {
      String name = klass.getName().toString();
      if (!name.startsWith("Lscript ") || !name.endsWith("/" + className)) continue;

      String file = name.substring("Lscript ".length(), name.length() - className.length() - 1);
      if (file.equals(wantedFile) || file.endsWith("/" + wantedFile)) return name.substring(1);
    }

    return null;
  }
}
