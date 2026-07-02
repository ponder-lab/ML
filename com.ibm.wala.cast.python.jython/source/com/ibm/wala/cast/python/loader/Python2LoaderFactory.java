package com.ibm.wala.cast.python.loader;

import com.ibm.wala.ipa.cha.IClassHierarchy;

public class Python2LoaderFactory extends PythonLoaderFactory {

  @Override
  protected PythonLoader makePythonLoader(IClassHierarchy cha) {
    return new Python2Loader(cha);
  }
}
