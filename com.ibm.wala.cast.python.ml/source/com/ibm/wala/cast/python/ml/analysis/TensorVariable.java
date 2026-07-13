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
package com.ibm.wala.cast.python.ml.analysis;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.ibm.wala.cast.python.ml.types.TensorOrigin;
import com.ibm.wala.cast.python.ml.types.TensorType;
import com.ibm.wala.fixpoint.IVariable;
import com.ibm.wala.util.collections.HashSetFactory;
import java.util.Collections;
import java.util.EnumSet;
import java.util.Set;
import java.util.stream.Collectors;

public class TensorVariable implements IVariable<TensorVariable> {
  private int graphNodeId = -1;
  private int orderNumber = -1;
  Set<TensorType> state = HashSetFactory.make();

  /**
   * The libraries whose operations produced this variable's value (wala/ML#724). Seeded from the
   * dispatched generator per dataflow source and unioned along the same edges as {@link #state}; a
   * merge of numpy-origin and TensorFlow-origin flows carries both constants. Unlike {@link
   * #state}, this set is never {@code null}: an empty set means no origin evidence reached the
   * variable.
   */
  Set<TensorOrigin> origins = EnumSet.noneOf(TensorOrigin.class);

  public String toFormattedString(TensorType.Format fmt) {
    switch (fmt) {
      case CString:
        return toCString(false);
      case MCString:
        return toCString(true);
      case MDString:
        return toMDString();
      case JsonSchema:
        {
          JsonElement elem = toJsonSchema();
          if (elem == null) {
            elem = new JsonObject();
          }
          return new Gson().toJson(elem);
        }
      default:
        throw new IllegalArgumentException("unknown format type: " + fmt);
    }
  }

  public JsonElement toJsonSchema() {
    if (state == null || state.isEmpty()) {
      return new JsonObject();
    } else {
      JsonArray arr = new JsonArray();
      state.stream().map(TensorType::toJsonSchema).forEach(x -> arr.add(x));
      if (state.size() == 1) {
        return arr.get(0);
      } else {
        final JsonObject obj = new JsonObject();
        obj.add("anyOf", arr);
        return obj;
      }
    }
  }

  public String toMDString() {
    if (state == null || state.isEmpty()) {
      return "?";
    }

    return state.stream().map(TensorType::toMDString).collect(Collectors.joining(" _or_ "));
  }

  public String toCString(boolean useMarkdown) {
    if (state == null || state.isEmpty()) {
      return null;
    }

    final String delim;
    if (useMarkdown) {
      delim = " _or_ ";
    } else {
      delim = " or ";
    }

    return state.stream().map(x -> x.toCString(useMarkdown)).collect(Collectors.joining(delim));
  }

  public Set<TensorType> getTypes() {
    return Collections.unmodifiableSet(state);
  }

  /**
   * Returns the libraries whose operations produced this variable's value (wala/ML#724).
   *
   * @return The producing libraries; empty when no origin evidence reached the variable, both
   *     constants when numpy-origin and TensorFlow-origin flows merged.
   */
  public Set<TensorOrigin> getOrigins() {
    return Collections.unmodifiableSet(origins);
  }

  @Override
  public int getGraphNodeId() {
    return graphNodeId;
  }

  @Override
  public void setGraphNodeId(int number) {
    graphNodeId = number;
  }

  @Override
  public int getOrderNumber() {
    return orderNumber;
  }

  @Override
  public void setOrderNumber(int i) {
    orderNumber = i;
  }

  @Override
  public void copyState(TensorVariable v) {
    this.state = v.state == null ? null : HashSetFactory.make(v.state);
    this.origins = EnumSet.noneOf(TensorOrigin.class);
    this.origins.addAll(v.origins);
  }

  @Override
  public String toString() {
    return state.toString();
  }
}
