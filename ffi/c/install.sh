#!/bin/sh
# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 TPT Solutions
#
# Installs the C library, its header and a pkg-config file under a prefix.
#
#   cargo build --release
#   sh install.sh [PREFIX]          # PREFIX defaults to /usr/local
#   DESTDIR=/tmp/stage sh install.sh /usr/local   # stage files for packaging
#
# The pkg-config file records PREFIX, not DESTDIR, so it stays correct after staging.
# LIB_DIR overrides where the built library is looked for (default target/release).
set -eu

prefix="${1:-/usr/local}"
dest="${DESTDIR:-}"
lib_dir="${LIB_DIR:-target/release}"

here=$(cd "$(dirname "$0")" && pwd)
cd "$here"

version=$(sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -n 1)
if [ -z "$version" ]; then
  echo "install.sh: could not read the version from Cargo.toml" >&2
  exit 1
fi

# The static archive is always built. The shared library name depends on the platform.
found_static=""
for name in libtpt_ml_proven.a libtpt_ml_proven.so libtpt_ml_proven.dylib; do
  if [ -f "$lib_dir/$name" ]; then
    found_static="$found_static $name"
  fi
done
case "$found_static" in
  *libtpt_ml_proven.a*) ;;
  *)
    echo "install.sh: $lib_dir/libtpt_ml_proven.a not found; run cargo build --release first" >&2
    exit 1
    ;;
esac

libdir="$dest$prefix/lib"
includedir="$dest$prefix/include"
pkgdir="$libdir/pkgconfig"
mkdir -p "$libdir" "$includedir" "$pkgdir"

for name in $found_static; do
  install -m 644 "$lib_dir/$name" "$libdir/$name"
done
install -m 644 include/tpt_ml_proven.h "$includedir/tpt_ml_proven.h"
sed -e "s|@PREFIX@|$prefix|" -e "s|@VERSION@|$version|" \
  pkgconfig/tpt-ml-proven.pc.in > "$pkgdir/tpt-ml-proven.pc"
chmod 644 "$pkgdir/tpt-ml-proven.pc"

echo "installed tpt-ml-proven $version under $dest$prefix"
