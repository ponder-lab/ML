/*
 * Copyright (c) 2013 IBM Corporation.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the Eclipse Public License v1.0
 * which accompanies this distribution, and is available at
 * http://www.eclipse.org/legal/epl-v10.html
 *
 * Contributors:
 *     IBM Corporation - initial API and implementation
 */
package com.ibm.wala.cast.python.ir;

import com.ibm.wala.cast.tree.CAstEntity;
import com.ibm.wala.cast.tree.CAstNode;
import com.ibm.wala.cast.tree.CAstSourcePositionMap;
import com.ibm.wala.cast.tree.CAstSymbol;
import com.ibm.wala.cast.tree.visit.CAstVisitor;
import com.ibm.wala.util.collections.HashMapFactory;
import com.ibm.wala.util.collections.MapUtil;
import com.ibm.wala.util.collections.Pair;
import java.util.Map;
import java.util.Set;

/**
 * Discovers which names declared by an {@link CAstEntity entity} are exposed, i.e., accessed by
 * nested functions: WALA's {@code com.ibm.wala.cast.ir.translator.ExposedNamesCollector} (1.8.0),
 * copied, which also descends into a comprehension. A comprehension's element and filter functions
 * are children of a {@link CAstNode#COMPREHENSION_EXPR}, and its filter functions are members of an
 * {@link CAstNode#EXPR_LIST}, kinds the WALA pass answers as handled without visiting them, so the
 * names those functions read from the enclosing function were never exposed there, and the
 * enclosing function wrote none of them to its lexical scope, unlike the names a lambda or a nested
 * function reads (<a href="https://github.com/wala/ML/issues/1066">wala/ML#1066</a>).
 *
 * <p>The pass is copied rather than extended because its context type is package-private, and a
 * subclass in WALA's package would split that package across bundles for a client loading each jar
 * as its own bundle.
 *
 * <p>TODO: Temporary. Remove this class, and {@link PythonCAstToIRTranslator}'s {@code translate}
 * override, once the WALA release Ariadne builds on lets a front end handle the node kinds it adds
 * (wala/ML#1066).
 */
public class PythonExposedNamesCollector
    extends CAstVisitor<PythonExposedNamesCollector.EntityContext> {

  /** names declared by each entity */
  private final Map<CAstEntity, Set<String>> entity2DeclaredNames = HashMapFactory.make();

  /** exposed names for each entity, updated as child entities are visited */
  private final Map<CAstEntity, Set<String>> entity2ExposedNames = HashMapFactory.make();

  /** exposed names for each entity which are written, updated as child entities are visited */
  private final Map<CAstEntity, Set<Pair<CAstEntity, String>>> entity2WrittenNames =
      HashMapFactory.make();

  public Map<CAstEntity, Set<String>> getEntity2ExposedNames() {
    return entity2ExposedNames;
  }

  public Map<CAstEntity, Set<Pair<CAstEntity, String>>> getEntity2WrittenNames() {
    return entity2WrittenNames;
  }

  /**
   * The context of the pass: the entity being visited.
   *
   * @param top The entity.
   */
  record EntityContext(CAstEntity top) implements CAstVisitor.Context {

    @Override
    public CAstSourcePositionMap getSourceMap() {
      return top.getSourceMap();
    }
  }

  /**
   * run the collector on an entity
   *
   * @param N the entity
   */
  public void run(CAstEntity N) {
    visitEntities(N, new EntityContext(N), this);
  }

  @Override
  protected EntityContext makeCodeContext(EntityContext context, CAstEntity n) {
    if (n.getKind() == CAstEntity.FUNCTION_ENTITY) {
      // need to handle arguments
      String[] argumentNames = n.getArgumentNames();
      for (String arg : argumentNames) {
        MapUtil.findOrCreateSet(entity2DeclaredNames, n).add(arg);
      }
    }
    return new EntityContext(n);
  }

  @Override
  protected void leaveDeclStmt(CAstNode n, EntityContext c, CAstVisitor<EntityContext> visitor) {
    CAstSymbol s = (CAstSymbol) n.getChild(0).getValue();
    String nm = s.name();
    MapUtil.findOrCreateSet(entity2DeclaredNames, c.top()).add(nm);
  }

  @Override
  protected void leaveFunctionStmt(
      CAstNode n, EntityContext c, CAstVisitor<EntityContext> visitor) {
    CAstEntity fn = (CAstEntity) n.getChild(0).getValue();
    String nm = fn.getName();
    MapUtil.findOrCreateSet(entity2DeclaredNames, c.top()).add(nm);
  }

  @Override
  protected void leaveClassStmt(CAstNode n, EntityContext c, CAstVisitor<EntityContext> visitor) {
    CAstEntity fn = (CAstEntity) n.getChild(0).getValue();
    String nm = fn.getName();
    MapUtil.findOrCreateSet(entity2DeclaredNames, c.top()).add(nm);
  }

  private void checkForLexicalAccess(Context c, String nm, boolean isWrite) {
    CAstEntity entity = c.top();
    final Set<String> entityNames = entity2DeclaredNames.get(entity);
    if (entityNames == null || !entityNames.contains(nm)) {
      CAstEntity declaringEntity = null;
      CAstEntity curEntity = getParent(entity);
      while (curEntity != null) {
        final Set<String> curEntityNames = entity2DeclaredNames.get(curEntity);
        if (curEntityNames != null && curEntityNames.contains(nm)) {
          declaringEntity = curEntity;
          break;
        } else {
          curEntity = getParent(curEntity);
        }
      }
      if (declaringEntity != null) {
        MapUtil.findOrCreateSet(entity2ExposedNames, declaringEntity).add(nm);
        if (isWrite) {
          MapUtil.findOrCreateSet(entity2WrittenNames, declaringEntity).add(Pair.make(entity, nm));
        }
      }
    }
  }

  @Override
  protected void leaveVar(CAstNode n, EntityContext c, CAstVisitor<EntityContext> visitor) {
    String nm = (String) n.getChild(0).getValue();
    checkForLexicalAccess(c, nm, false);
  }

  @Override
  protected void leaveVarAssignOp(
      CAstNode n,
      CAstNode v,
      CAstNode a,
      boolean pre,
      EntityContext c,
      CAstVisitor<EntityContext> visitor) {
    checkForLexicalAccess(c, (String) n.getChild(0).getValue(), true);
  }

  @Override
  protected void leaveVarAssign(
      CAstNode n, CAstNode v, CAstNode a, EntityContext c, CAstVisitor<EntityContext> visitor) {
    checkForLexicalAccess(c, (String) n.getChild(0).getValue(), true);
  }

  @Override
  protected boolean doVisit(CAstNode n, EntityContext context, CAstVisitor<EntityContext> visitor) {
    if (n.getKind() == CAstNode.COMPREHENSION_EXPR) {
      visitComprehensionChildren(n, context, visitor);
      return true;
    }
    // assume unknown node types don't do anything relevant to exposed names.
    // override if this is untrue
    return true;
  }

  /**
   * Visits a comprehension's children: its collection, its element function, and the lists of its
   * iterables and of its filter functions. A list is itself a kind this pass does not otherwise
   * descend into, so its members are visited here.
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
        visitComprehensionChildren(child, context, visitor);
      else visitor.visit(child, context, visitor);
    }
  }

  @Override
  protected boolean doVisitAssignNodes(
      CAstNode n,
      EntityContext context,
      CAstNode v,
      CAstNode a,
      CAstVisitor<EntityContext> visitor) {
    // assume unknown node types don't do anything relevant to exposed names.
    // override if this is untrue
    return true;
  }
}
