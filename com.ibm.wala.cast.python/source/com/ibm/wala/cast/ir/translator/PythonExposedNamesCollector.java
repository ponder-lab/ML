package com.ibm.wala.cast.ir.translator;

import com.ibm.wala.cast.tree.CAstNode;
import com.ibm.wala.cast.tree.visit.CAstVisitor;

/**
 * The exposed-names pass for Python, which also descends into a comprehension. A comprehension's
 * element and filter functions are children of a {@link CAstNode#COMPREHENSION_EXPR}, a kind the
 * base pass answers as handled without visiting it, so the names those functions read from the
 * enclosing function were never exposed there, and the enclosing function wrote none of them to its
 * lexical scope, unlike the names a lambda or a nested function reads (<a
 * href="https://github.com/wala/ML/issues/1066">wala/ML#1066</a>).
 *
 * <p>The lists of a comprehension's iterables and filter functions are {@link CAstNode#EXPR_LIST}s,
 * which the base pass does not descend into either, so the pass descends into those of a
 * comprehension too. It lives in the base pass's package because the pass's context type is
 * package-private.
 *
 * <p>TODO: Temporary. Remove this class, its package, and {@code PythonCAstToIRTranslator}'s {@code
 * translate} override once the WALA release Ariadne builds on visits the children of node kinds the
 * base pass does not know, or lets a front end supply its own pass (<a
 * href="https://github.com/wala/ML/issues/1066">wala/ML#1066</a>).
 */
public class PythonExposedNamesCollector extends ExposedNamesCollector {

  @Override
  protected boolean doVisit(CAstNode n, EntityContext context, CAstVisitor<EntityContext> visitor) {
    if (n.getKind() == CAstNode.COMPREHENSION_EXPR) {
      this.visitComprehensionChildren(n, context, visitor);
      return true;
    }
    return super.doVisit(n, context, visitor);
  }

  /**
   * Visits a comprehension's children: its collection, its element function, and the lists of its
   * iterables and of its filter functions. A list is itself a kind the base pass does not descend
   * into, so its members are visited here.
   *
   * @param n The comprehension, or one of its lists.
   * @param context The pass's context.
   * @param visitor The pass.
   */
  private void visitComprehensionChildren(
      CAstNode n, EntityContext context, CAstVisitor<EntityContext> visitor) {
    for (int i = 0; i < n.getChildCount(); i++) {
      CAstNode child = n.getChild(i);
      if (child.getKind() == CAstNode.EXPR_LIST)
        this.visitComprehensionChildren(child, context, visitor);
      else visitor.visit(child, context, visitor);
    }
  }
}
