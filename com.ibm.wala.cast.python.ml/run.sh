#!/bin/bash

ME=`realpath $0`
DIR=`dirname $ME`

# Resolve the `java` executable. Prefer `$JAVA_HOME/bin/java` when `JAVA_HOME` is
# set (the common case for IDE-launched LSP); fall back to `java` on `PATH` when
# it isn't. Pre-fix the script unconditionally referenced `$JAVA_HOME/bin/java`,
# which expanded to `/bin/java` if `JAVA_HOME` was unset (almost always fails).
# See https://github.com/wala/ML/issues/542.
if [ -n "$JAVA_HOME" ] && [ -x "$JAVA_HOME/bin/java" ]; then
  JAVA="$JAVA_HOME/bin/java"
elif command -v java >/dev/null 2>&1; then
  JAVA=java
else
  echo "no \`java\` executable found: set \`JAVA_HOME\` to a JDK install (with \`bin/java\` present) or add \`java\` to PATH." >&2
  exit 1
fi

# Locate the fat classifier JAR (built by `mvn package` per
# https://github.com/wala/ML/issues/525). Glob avoids version-pinning
# brittleness—the previous hardcoded `0.40.0-SNAPSHOT` path fell ~5 minor
# versions behind the project's actual `pom.xml` version.
shopt -s nullglob
JARS=("$DIR"/target/com.ibm.wala.cast.python.ml-*-fat.jar)
shopt -u nullglob

if [ ${#JARS[@]} -eq 0 ]; then
  echo "fat JAR not found in $DIR/target/. Run \`mvn -f \"$DIR/../pom.xml\" -pl com.ibm.wala.cast.python.ml -am clean package -DskipTests\` first." >&2
  exit 1
fi

if [ ${#JARS[@]} -gt 1 ]; then
  echo "Multiple fat JARs in $DIR/target/: ${JARS[*]}. Remove stale ones or run a \`clean package\` to get a single matching jar." >&2
  exit 1
fi

JAR="${JARS[0]}"

cat -u | tee -a /tmp/lsp.in.log | "$JAVA" -Xdebug -Xrunjdwp:transport=dt_socket,address=127.0.0.1:6660,server=y,suspend=n -jar "$JAR" --mode stdio | tee -a /tmp/lsp.out.log
