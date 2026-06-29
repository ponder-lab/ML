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
package com.ibm.wala.cast.python.loader;

import static java.util.logging.Level.FINE;
import static java.util.logging.Level.WARNING;

import com.ibm.wala.cast.ir.translator.ConstantFoldingRewriter;
import com.ibm.wala.cast.ir.translator.RewritingTranslatorToCAst;
import com.ibm.wala.cast.ir.translator.TranslatorToCAst;
import com.ibm.wala.cast.python.parser.PythonModuleParser;
import com.ibm.wala.cast.python.util.Python3Interpreter;
import com.ibm.wala.cast.tree.CAst;
import com.ibm.wala.cast.tree.CAstEntity;
import com.ibm.wala.cast.tree.impl.CAstOperator;
import com.ibm.wala.cast.tree.rewrite.AstConstantFolder;
import com.ibm.wala.cast.tree.rewrite.CAstBasicRewriter.NoKey;
import com.ibm.wala.cast.tree.rewrite.CAstBasicRewriter.NonCopyingContext;
import com.ibm.wala.cast.tree.rewrite.CAstRewriterFactory;
import com.ibm.wala.cast.tree.rewrite.PatternBasedRewriter;
import com.ibm.wala.cast.util.CAstPattern.Segments;
import com.ibm.wala.classLoader.IClassLoader;
import com.ibm.wala.classLoader.Module;
import com.ibm.wala.classLoader.ModuleEntry;
import com.ibm.wala.classLoader.SourceModule;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Logger;
import org.python.core.PyObject;
import org.python.core.PySyntaxError;
import org.python.core.PyUnicode;
import org.python.util.PythonInterpreter;

public class Python3Loader extends PythonLoader {

  private static final Logger logger = Logger.getLogger(Python3Loader.class.getName());

  /**
   * Memoizes whether the "parser unavailable" warning has already been emitted from constant
   * folding. When the embedded interpreter compiles a non-constant expression it hits a parse
   * error, and Jython's {@link org.python.core.ParserFacade#fixParseError} references {@code jline}
   * to format it. Under a minimal runtime classpath (the published {@code
   * com.ibm.wala.cast.python.ml} fat-jar, which ships no {@code jline} classes) that reference
   * throws {@link NoClassDefFoundError} instead of producing a {@link PySyntaxError}. Folding
   * catches it and degrades (skips that expression), but every non-constant subexpression would
   * otherwise re-log; this flag keeps it to one WARNING, FINE thereafter. See <a
   * href="https://github.com/wala/ML/issues/631">wala/ML#631</a>.
   */
  private static final AtomicBoolean parserUnavailableWarned = new AtomicBoolean(false);

  public Python3Loader(IClassHierarchy cha, IClassLoader parent, List<File> pythonPath) {
    super(cha, parent, pythonPath);
  }

  public Python3Loader(IClassHierarchy cha, List<File> pythonPath) {
    super(cha, pythonPath);
  }

  @Override
  protected TranslatorToCAst getTranslatorToCAst(CAst ast, ModuleEntry M, List<Module> allModules)
      throws IOException {
    RewritingTranslatorToCAst x =
        new RewritingTranslatorToCAst(
            M,
            new PythonModuleParser(
                (SourceModule) M, typeDictionary, allModules, this.getPythonPath()) {
              @Override
              public CAstEntity translateToCAst() throws Error, IOException {
                CAstEntity ce = super.translateToCAst();
                return new AstConstantFolder().fold(ce);
              }
            });

    x.addRewriter(
        new CAstRewriterFactory<NonCopyingContext, NoKey>() {
          @Override
          public PatternBasedRewriter createCAstRewriter(CAst ast) {
            return new PatternBasedRewriter(
                ast,
                sliceAssign,
                (Segments s) -> {
                  return rewriteSubscriptAssign(s);
                });
          }
        },
        false);

    x.addRewriter(
        new CAstRewriterFactory<NonCopyingContext, NoKey>() {
          @Override
          public PatternBasedRewriter createCAstRewriter(CAst ast) {
            return new PatternBasedRewriter(
                ast,
                sliceAssignOp,
                (Segments s) -> {
                  return rewriteSubscriptAssignOp(s);
                });
          }
        },
        false);

    x.addRewriter(
        new CAstRewriterFactory<NonCopyingContext, NoKey>() {
          @Override
          public ConstantFoldingRewriter createCAstRewriter(CAst ast) {
            return new ConstantFoldingRewriter(ast) {
              @Override
              protected Object eval(CAstOperator op, Object lhs, Object rhs) {
                String s = lhs + " " + op.getValue() + " " + rhs;

                PythonInterpreter ip = Python3Interpreter.getInterp();
                if (ip == null) {
                  // Jython init failed (memoized in Python3Interpreter). Skip constant folding
                  // for this expression; analysis remains correct, just less precise. Don't log
                  // an "Evaluating:" entry — nothing is actually evaluated, and the underlying
                  // init failure was already announced from getInterp().
                  return null;
                }
                logger.info(() -> "Evaluating: " + s);

                // Use the Python interpreter to evaluate the expression.
                PyUnicode unicode = new PyUnicode(s);
                PyObject x;

                try {
                  x = ip.eval(unicode);
                } catch (PySyntaxError e) {
                  // Handle syntax errors gracefully.
                  logger.log(WARNING, e, () -> "Syntax error in expression: " + unicode);
                  return null;
                } catch (LinkageError e) {
                  // A non-constant expression hits a parse error during eval, and Jython's
                  // ParserFacade.fixParseError references jline to format it. Under a minimal
                  // runtime classpath (the published fat-jar, which ships no jline classes) that
                  // reference throws NoClassDefFoundError (a LinkageError) before a PySyntaxError
                  // is
                  // produced, so the catch above never sees it. Treat it like the syntax error it
                  // stands in for: skip folding this expression. Catching LinkageError (not Error)
                  // keeps serious failures (OOM, stack overflow) propagating. The first occurrence
                  // is logged at WARNING so operators see folding is degraded; the rest at FINE to
                  // avoid one entry per non-constant subexpression. See wala/ML#631.
                  if (parserUnavailableWarned.compareAndSet(false, true))
                    logger.log(
                        WARNING,
                        e,
                        () ->
                            "Cannot format a parse error during constant folding (jline missing"
                                + " from the runtime classpath); skipping folding for this and any"
                                + " subsequent non-constant expression. First skipped: "
                                + unicode);
                  else
                    logger.log(FINE, e, () -> "Skipping folding (parser unavailable): " + unicode);
                  return null;
                }

                if (x.isNumberType()) {
                  // If the result is a number, return its integer value.
                  logger.info(() -> s + " -> " + x.asInt());
                  return x.asInt();
                }
                return null;
              }
            };
          }
        },
        false);
    return x;
  }
}
