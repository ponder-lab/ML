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
package com.ibm.wala.cast.python.ipa.summaries;

import com.ibm.wala.cast.python.loader.StarFormalDeclaration;
import com.ibm.wala.classLoader.IClass;
import com.ibm.wala.ipa.summaries.MethodSummary;
import com.ibm.wala.types.MethodReference;

/**
 * The synthesized constructor of a source-defined Python class (see {@code
 * PythonConstructorTargetSelector}): allocates the instance, wires the method trampolines, and
 * forwards to {@code __init__}. Distinguished from other {@link PythonSummarizedFunction}s so the
 * engine's context selector can dispatch the internal {@code __init__} call in the constructor's
 * own calling context — the constructor body has a single {@code __init__} call site, so a plain
 * call-string context there collapses every construction of the class into one {@code __init__}
 * context, unioning the argument values across construction sites (<a
 * href="https://github.com/wala/ML/issues/671">wala/ML#671</a>).
 *
 * @author <a href="mailto:khatchad@hunter.cuny.edu">Raffi Khatchadourian</a>
 */
public class PythonConstructorFunction extends PythonSummarizedFunction
    implements StarFormalDeclaration {

  /**
   * The wrapped {@code __init__}'s defaulted-parameter count; see {@link
   * #getNumberOfDefaultParameters()}.
   */
  private final int initDefaultParameters;

  /**
   * The wrapped {@code __init__}'s count of trailing formals that cannot take a positional default;
   * see {@link #getNumberOfTrailingNonDefaultableParameters()}.
   */
  private final int initTrailingNonDefaultableParameters;

  /**
   * The index of the constructor's own {@code *args} formal among its arguments, or {@code -1}: the
   * wrapped {@code __init__}'s shifted by one, since constructor argument {@code j} is {@code
   * __init__} argument {@code j + 1}.
   */
  private final int varargsParameter;

  /**
   * The index of the constructor's own {@code **kwargs} formal, shifted likewise, or {@code -1}.
   */
  private final int keywordsParameter;

  /**
   * Constructs a {@link PythonConstructorFunction}.
   *
   * @param ref The constructor's method reference.
   * @param summary The synthesized constructor body.
   * @param declaringClass The class being constructed.
   * @param initDefaultParameters The wrapped {@code __init__}'s defaulted-parameter count.
   * @param initTrailingNonDefaultableParameters The wrapped {@code __init__}'s count of trailing
   *     formals that cannot take a positional default.
   * @param varargsParameter The constructor's own {@code *args} formal index, or {@code -1}.
   * @param keywordsParameter The constructor's own {@code **kwargs} formal index, or {@code -1}.
   */
  public PythonConstructorFunction(
      MethodReference ref,
      MethodSummary summary,
      IClass declaringClass,
      int initDefaultParameters,
      int initTrailingNonDefaultableParameters,
      int varargsParameter,
      int keywordsParameter) {
    super(ref, summary, declaringClass);
    this.initDefaultParameters = initDefaultParameters;
    this.initTrailingNonDefaultableParameters = initTrailingNonDefaultableParameters;
    this.varargsParameter = varargsParameter;
    this.keywordsParameter = keywordsParameter;
  }

  /**
   * The constructor's {@code *args} formal, so a class call's positional arguments past {@code
   * __init__}'s formals are packed onto it and forwarded to {@code __init__}'s {@code *args}
   * (wala/ML#188). Without it the constructor declared no star formal and those arguments were
   * dropped at the class call.
   *
   * @return The index, or {@code -1} when {@code __init__} declares no {@code *args}.
   */
  @Override
  public int getVarargsParameter() {
    return this.varargsParameter;
  }

  /**
   * The constructor's {@code **kwargs} formal, so a class call's keywords naming no formal are
   * packed onto it and forwarded to {@code __init__}'s {@code **kwargs} (wala/ML#997); likewise
   * dropped before.
   *
   * @return The index, or {@code -1} when {@code __init__} declares no {@code **kwargs}.
   */
  @Override
  public int getKeywordsParameter() {
    return this.keywordsParameter;
  }

  /**
   * The constructor's trailing formals mirror the wrapped {@code __init__}'s parameters, so its
   * defaulted-parameter count carries over: an instantiation that leaves them unpassed binds them
   * from {@code __init__}'s default globals (wala/ML#762).
   *
   * @return The wrapped {@code __init__}'s defaulted-parameter count.
   */
  @Override
  public int getNumberOfDefaultParameters() {
    return this.initDefaultParameters;
  }

  /**
   * The constructor's trailing formals mirror the wrapped {@code __init__}'s parameters, so its
   * count of formals that cannot take a positional default carries over too. Without it the
   * defaulted range is located by counting back from the end of a formal list that includes the
   * {@code **kwargs} slot, which binds every default one parameter to the right (wala/ML#843).
   *
   * @return The wrapped {@code __init__}'s count.
   */
  @Override
  public int getNumberOfTrailingNonDefaultableParameters() {
    return this.initTrailingNonDefaultableParameters;
  }
}
