package com.ibm.wala.cast.python.ipa.callgraph;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.object;

import com.ibm.wala.cast.ir.ssa.AstInstructionFactory;
import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.PythonLoader;
import com.ibm.wala.cast.python.loader.PythonLoader.DynamicMethodBody;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.python.util.Util.DecoratorCall;
import com.ibm.wala.classLoader.IClass;
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
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * An {@link Entrypoint} of a <a href="http://pytest.org">Pytest</a> test case.
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class PytesttEntrypoint extends DefaultEntrypoint {

  private static final Logger LOGGER = Logger.getLogger(PytesttEntrypoint.class.getName());

  /**
   * The decorator whose arguments supply a parameterized test method's parameter, one per test
   * case: {@code absl.testing.parameterized}'s {@code parameters}. Matched against the mined
   * decorator name by suffix, since the site may spell it {@code parameterized.parameters} or fully
   * qualified. Recognition is deliberately name-gated (wala/ML#868): binding any decorator that
   * happens to carry mineable class-name arguments would inject values through decorators with
   * entirely different runtime semantics.
   */
  private static final String PARAMETERIZED_PARAMETERS_DECORATOR = "parameterized.parameters";

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

    // A parameter whose only supplier is a `parameterized.parameters` decorator argument is
    // otherwise invisible: decoration is never applied in IR, so the decorator's per-argument
    // invocations of the test method do not exist anywhere the analysis can see (wala/ML#868).
    // Bind the parameter here, where the fake root already reads script globals to reach the
    // method itself.
    List<Integer> bindings = parameterizedArgumentBindings(m);

    if (!bindings.isEmpty()) {
      // One invocation PER ARGUMENT, deliberately, rather than one invocation with the union:
      // the runtime runs N separate test cases, one per argument, and separate invocations give
      // each parameterization its own context, so the A-parameterization's receivers resolve to A
      // alone. A single unioned binding would reintroduce, one level up, exactly the receiver
      // blur this issue is about.
      PythonInvokeInstruction last = null;

      for (int binding : bindings) {
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
   * Emits a global read per resolvable class-name argument of this method's {@code
   * parameterized.parameters} decorator, returning the read values in argument order (wala/ML#868).
   * The read reaches the defining script's own class object at its own allocation site, the same
   * global-read idiom the entrypoint's parameter zero uses, so class identity and attributes are
   * preserved.
   *
   * <p>Binding applies only when the method is a {@link DynamicMethodBody} carrying a recognized
   * decorator (see {@link #PARAMETERIZED_PARAMETERS_DECORATOR}) and takes exactly one parameter
   * beyond the function object and the receiver: a multi-parameter parameterization supplies
   * tuples, which mine as unmineable and must not be guessed at. Per-argument residuals, each
   * logged at FINE and each leaving that argument's test case exactly as unmodeled as it is today:
   * an explicit {@code None} (a null-constant binding is a stated follow-up; the subject's
   * guard-on-{@code None} arms contribute no evidence until then), an unmineable argument, and a
   * name that does not resolve to exactly one script-defined class. Module-attribute tokens that
   * are not script-defined classes (e.g. NumPy dtype objects) do not resolve under this resolver BY
   * DESIGN: this version binds script-defined classes only, so those parameters keep today's
   * behavior, and widening to summary-module tokens is a deliberate later choice rather than a side
   * effect.
   *
   * @param m The root method being built.
   * @return The value numbers holding each resolved class object, empty when nothing binds.
   */
  private List<Integer> parameterizedArgumentBindings(AbstractRootMethod m) {
    List<Integer> ret = new ArrayList<>();

    // Method-form only, with exactly one parameter beyond the function object and the receiver:
    // the fake-root invocation's slot 1 maps through the bound-method trampoline (which supplies
    // the receiver itself) to that parameter. A multi-parameter parameterization supplies tuples,
    // which mine as unmineable and must not be guessed at.
    if (!(getMethod().getDeclaringClass() instanceof DynamicMethodBody)) return ret;
    if (getNumberOfParameters() != 3) return ret;

    DynamicMethodBody body = (DynamicMethodBody) getMethod().getDeclaringClass();
    String methodDeclaringClassName = getMethod().getDeclaringClass().getName().toString();
    String script = methodDeclaringClassName.substring(1, methodDeclaringClassName.indexOf('/'));

    for (DecoratorCall decoratorCall : body.getDecoratorCalls()) {
      String name = decoratorCall.name();
      if (!name.equals(PARAMETERIZED_PARAMETERS_DECORATOR)
          && !name.endsWith("." + PARAMETERIZED_PARAMETERS_DECORATOR)) continue;

      for (String argument : decoratorCall.argumentNames()) {
        String globalName = resolveClassGlobal(script, argument);

        if (globalName == null) {
          LOGGER.fine(
              () ->
                  "Parameterized argument: "
                      + argument
                      + " of: "
                      + methodDeclaringClassName
                      + " does not bind; its test case stays unmodeled (wala/ML#868).");
          continue;
        }

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
    }

    return ret;
  }

  /**
   * Resolves a mined decorator-argument name to the defining script's global name for the class
   * object, or {@code null} when it does not resolve to exactly one script-defined class. A bare
   * name resolves in the method's own script first, then by a unique cross-script match (covering
   * {@code from module import Class}, whose class is defined elsewhere); a dotted name resolves by
   * a unique match among classes of that name whose defining script's file name is the qualifier's
   * last segment. Ambiguity declines rather than guessing: binding the wrong class would be
   * confidently wrong, which is worse than today's unbound state.
   *
   * @param script The entrypoint method's own script (e.g. {@code script test_x.py}).
   * @param argument The mined argument name, bare or dotted; {@code None} and the unmineable marker
   *     never resolve.
   * @return The global's name (e.g. {@code script mod.py/A}), or {@code null}.
   */
  private String resolveClassGlobal(String script, String argument) {
    if (argument.indexOf('.') < 0) {
      String candidate = script + "/" + argument;

      if (getCha()
              .lookupClass(TypeReference.findOrCreate(PythonTypes.pythonLoader, "L" + candidate))
          != null) return candidate;

      return uniqueScriptClass(argument, null);
    }

    int dot = argument.lastIndexOf('.');
    String qualifier = argument.substring(0, dot);
    String className = argument.substring(dot + 1);
    String wantedFile = qualifier.substring(qualifier.lastIndexOf('.') + 1) + ".py";

    return uniqueScriptClass(className, wantedFile);
  }

  /**
   * Finds the single script-defined class of the given name, optionally filtered by its defining
   * script's file name, returning its global name or {@code null} on zero or several matches.
   *
   * @param className The class's simple name.
   * @param wantedFile The defining script's file name to require, or {@code null} for any.
   * @return The unique match's global name, or {@code null}.
   */
  private String uniqueScriptClass(String className, String wantedFile) {
    String match = null;

    for (IClass klass : getCha()) {
      String name = klass.getName().toString();
      if (!name.startsWith("Lscript ") || !name.endsWith("/" + className)) continue;

      String file = name.substring("Lscript ".length(), name.length() - className.length() - 1);
      if (wantedFile != null && !file.equals(wantedFile) && !file.endsWith("/" + wantedFile))
        continue;
      // A nested class's "file" segment carries the outer class's path components and cannot end
      // with a Python file name; requiring the segment to look like a file keeps outer-class
      // members from matching a bare-name scan.
      if (wantedFile == null && !file.endsWith(".py")) continue;

      if (match != null) {
        LOGGER.fine(
            () ->
                "Parameterized argument class name: "
                    + className
                    + " is ambiguous across scripts; declining to bind (wala/ML#868).");
        return null;
      }

      match = name.substring(1);
    }

    return match;
  }
}
