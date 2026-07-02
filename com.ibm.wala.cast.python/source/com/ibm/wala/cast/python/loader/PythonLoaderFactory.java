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

import com.ibm.wala.cast.loader.SingleClassLoaderFactory;
import com.ibm.wala.cast.python.types.PythonTypes;
import com.ibm.wala.classLoader.IClassLoader;
import com.ibm.wala.ipa.cha.IClassHierarchy;
import com.ibm.wala.ipa.summaries.XMLMethodSummaryReader;
import com.ibm.wala.types.ClassLoaderReference;
import java.util.ArrayList;
import java.util.List;

public abstract class PythonLoaderFactory extends SingleClassLoaderFactory {

  /**
   * Summary readers whose {@code <class name="..." super="...">} declarations should be
   * materialized as class shells in the loader before source translation (wala/ML#118). Registered
   * by the engine via {@link #addSummaryClassShells} before the class hierarchy is built.
   */
  private final List<XMLMethodSummaryReader> summaryClassShellReaders = new ArrayList<>();

  /**
   * Registers a summary reader whose {@code <class super="...">} declarations should become class
   * shells in the loader this factory creates. Must be called before the class hierarchy is built,
   * since the shells are materialized when the loader initializes.
   *
   * @param summaries the reader whose class-shell declarations to materialize
   */
  public void addSummaryClassShells(XMLMethodSummaryReader summaries) {
    summaryClassShellReaders.add(summaries);
  }

  /**
   * {@inheritDoc}
   *
   * <p>Hands the registered summary class-shell readers to the loader, so {@link PythonLoader#init}
   * can materialize the shells before source translation.
   */
  @Override
  protected final IClassLoader makeTheLoader(IClassHierarchy cha) {
    PythonLoader loader = makePythonLoader(cha);
    loader.setSummaryClassShellReaders(summaryClassShellReaders);
    return loader;
  }

  /**
   * Creates the concrete {@link PythonLoader} for this factory's Python front end.
   *
   * @param cha the class hierarchy under construction
   * @return the loader
   */
  protected abstract PythonLoader makePythonLoader(IClassHierarchy cha);

  @Override
  public ClassLoaderReference getTheReference() {
    return PythonTypes.pythonLoader;
  }
}
