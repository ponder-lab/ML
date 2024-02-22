package com.ibm.wala.cast.python.ipa.callgraph;

import static com.ibm.wala.cast.python.types.PythonTypes.Root;
import static com.ibm.wala.cast.python.types.PythonTypes.object;
import static com.ibm.wala.cast.python.types.Util.getFilename;
import static java.util.Objects.requireNonNull;

import com.ibm.wala.cast.ir.ssa.AstInstructionFactory;
import com.ibm.wala.cast.loader.DynamicCallSiteReference;
import com.ibm.wala.cast.python.ir.PythonLanguage;
import com.ibm.wala.cast.python.loader.PythonLoader;
import com.ibm.wala.cast.python.loader.PythonLoader.DynamicMethodBody;
import com.ibm.wala.cast.python.loader.PythonLoader.PythonClass;
import com.ibm.wala.cast.python.ssa.PythonInvokeInstruction;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.cast.types.AstMethodReference;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.client.AbstractAnalysisEngine.EntrypointBuilder;
import com.ibm.wala.core.util.strings.Atom;
import com.ibm.wala.ipa.callgraph.Entrypoint;
import com.ibm.wala.ipa.callgraph.impl.AbstractRootMethod;
import com.ibm.wala.ipa.callgraph.impl.DefaultEntrypoint;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ssa.SSAAbstractInvokeInstruction;
import com.ibm.wala.types.FieldReference;
import com.ibm.wala.types.MethodReference;
import com.ibm.wala.types.TypeName;
import com.ibm.wala.types.TypeReference;
import com.ibm.wala.util.collections.HashSetFactory;
import com.ibm.wala.util.collections.Pair;
import java.util.HashSet;
import java.util.logging.Logger;

/**
 * This class represents entry points ({@link Entrypoint})s of Pytest test functions. Pytest test
 * functions are those invoked by the pytest framework reflectively. The entry points can be used to
 * specify entry points of a call graph.
 */
public class PytestEntrypointBuilder implements EntrypointBuilder {

  private final class PytesttEntrypoint extends DefaultEntrypoint {

    private PytesttEntrypoint(MethodReference method, IClassHierarchy cha) {
      super(method, cha);
    }

    @Override
    public SSAAbstractInvokeInstruction addCall(AbstractRootMethod m) {
      int paramValues[];
      paramValues = new int[getNumberOfParameters()];

      for (int j = 0; j < paramValues.length; j++) {
        AstInstructionFactory insts = PythonLanguage.Python.instructionFactory();

        if (j == 0 && getMethod().getDeclaringClass().getName().toString().contains("/")) {
          int v = m.nextLocal++;
          paramValues[j] = v;

          if (getMethod().getDeclaringClass() instanceof PythonLoader.DynamicMethodBody) {
            FieldReference global =
                FieldReference.findOrCreate(
                    PythonTypes.Root,
                    Atom.findOrCreateUnicodeAtom(
                        "global "
                            + getMethod()
                                .getDeclaringClass()
                                .getName()
                                .toString()
                                .substring(
                                    1,
                                    getMethod()
                                        .getDeclaringClass()
                                        .getName()
                                        .toString()
                                        .lastIndexOf('/'))),
                    PythonTypes.Root);

            int idx = m.statements.size();
            int cls = m.nextLocal++;
            int obj = m.nextLocal++;
            m.statements.add(insts.GlobalRead(m.statements.size(), cls, global));
            idx = m.statements.size();

            @SuppressWarnings("unchecked")
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
            String method = getMethod().getDeclaringClass().getName().toString();
            String field = method.substring(method.lastIndexOf('/') + 1);
            FieldReference f =
                FieldReference.findOrCreate(
                    PythonTypes.Root, Atom.findOrCreateUnicodeAtom(field), PythonTypes.Root);

            m.statements.add(insts.GetInstruction(idx, v, obj, f));
          } else {
            FieldReference global =
                FieldReference.findOrCreate(
                    PythonTypes.Root,
                    Atom.findOrCreateUnicodeAtom(
                        "global "
                            + getMethod().getDeclaringClass().getName().toString().substring(1)),
                    PythonTypes.Root);

            m.statements.add(insts.GlobalRead(m.statements.size(), v, global));
          }
        } else {
          paramValues[j] = makeArgument(m, j);
        }

        if (paramValues[j] == -1) {
          // there was a problem
          return null;
        }

        TypeReference x[] = getParameterTypes(j);

        if (x.length == 1 && x[0].equals(object))
          m.statements.add(
              insts.PutInstruction(
                  m.statements.size(),
                  paramValues[j],
                  paramValues[j],
                  FieldReference.findOrCreate(
                      object, Atom.findOrCreateUnicodeAtom("pytest"), Root)));
      }

      int pc = m.statements.size();

      @SuppressWarnings("unchecked")
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
  }

  private static final Logger logger = Logger.getLogger(PytestEntrypointBuilder.class.getName());

  /**
   * Construct pytest entrypoints for all the pytest test functions in the given scope.
   *
   * @throws NullPointerException If the given {@link IClassHierarchy} is null.
   */
  @Override
  public Iterable<Entrypoint> createEntrypoints(IClassHierarchy cha) {
    requireNonNull(cha);

    final HashSet<Entrypoint> result = HashSetFactory.make();

    for (IClass klass : cha) {
      // if the class is a pytest test case,
      if (isPytestCase(klass)) {
        logger.fine(() -> "Pytest case: " + klass + ".");

        MethodReference methodReference =
            MethodReference.findOrCreate(klass.getReference(), AstMethodReference.fnSelector);

        result.add(new PytesttEntrypoint(methodReference, cha));

        logger.fine(() -> "Adding test method as entry point: " + methodReference.getName() + ".");
      }
    }

    return result::iterator;
  }

  /**
   * Check if the given class is a Pytest test class according to: https://bit.ly/3wj8nPY.
   *
   * @throws NullPointerException If the given {@link IClass} is null.
   * @see https://bit.ly/3wj8nPY.
   */
  public static boolean isPytestCase(IClass klass) {
    requireNonNull(klass);

    final TypeName typeName = klass.getReference().getName();

    if (typeName.toString().startsWith("Lscript ")) {
      final String fileName = getFilename(typeName);
      final Atom className = typeName.getClassName();

      // In Ariadne, a script is an invokable entity like a function.
      final boolean script = className.toString().endsWith(".py");

      if (!script // it's not an invokable script.
          && (fileName.startsWith("test_")
              || fileName.endsWith("_test")) // we're inside of a "test" file,
          && !(klass instanceof PythonClass)) { // classes aren't entrypoints.
        if (klass instanceof DynamicMethodBody) {
          // It's a method. In Ariadne, functions are also classes.
          DynamicMethodBody dmb = (DynamicMethodBody) klass;
          IClass container = dmb.getContainer();
          String containerName = container.getReference().getName().getClassName().toString();

          if (containerName.startsWith("Test") && container instanceof PythonClass) {
            // It's a test class.
            PythonClass containerClass = (PythonClass) container;

            final boolean hasCtor =
                containerClass.getMethodReferences().stream()
                    .anyMatch(
                        mr -> {
                          return mr.getName().toString().equals("__init__");
                        });

            // Test classes can't have constructors.
            if (!hasCtor) {
              // In Ariadne, methods are modeled as classes. Thus, a class name in this case is the
              // method name.
              String methodName = className.toString();

              // If the method starts with "test."
              if (methodName.startsWith("test")) return true;
            }
          }
        } else if (className.toString().startsWith("test")) return true; // It's a function.
      }
    }

    return false;
  }
}
