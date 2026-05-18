#!/bin/bash

ME=`realpath $0`
DIR=`dirname $ME`

# Locate the fat classifier JAR (built by `mvn package` per
# https://github.com/wala/ML/issues/525). Glob avoids version-pinning
# brittleness—the previous hardcoded `0.40.0-SNAPSHOT` path fell ~5 minor
# versions behind the project's actual `pom.xml` version.
shopt -s nullglob
JARS=("$DIR"/target/com.ibm.wala.cast.python.ml-*-fat.jar)
shopt -u nullglob

if [ ${#JARS[@]} -eq 0 ]; then
  echo "fat JAR not found in $DIR/target/. Run \`mvn -pl com.ibm.wala.cast.python.ml -am clean package -DskipTests\` first." >&2
  exit 1
fi

if [ ${#JARS[@]} -gt 1 ]; then
  echo "Multiple fat JARs in $DIR/target/: ${JARS[*]}. Remove stale ones or run a \`clean package\` to get a single matching jar." >&2
  exit 1
fi

JAR="${JARS[0]}"

cat -u | tee -a /tmp/lsp.in.log | $JAVA_HOME/bin/java -Xdebug -Xrunjdwp:transport=dt_socket,address=127.0.0.1:6660,server=y,suspend=n -jar "$JAR" --mode stdio | tee -a /tmp/lsp.out.log
