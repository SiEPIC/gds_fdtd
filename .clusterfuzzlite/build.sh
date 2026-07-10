#!/bin/bash -eu
# Build the atheris fuzz targets and their seed corpora for ClusterFuzzLite.

pip3 install .

for target in "$SRC/gds_fdtd/fuzz"/fuzz_*.py; do
  name="$(basename "$target" .py)"
  compile_python_fuzzer "$target"
done

# Seed corpora: start fuzzing from valid inputs so mutations reach the
# meaningful parsing paths quickly.
zip -j "$OUT/fuzz_technology_seed_corpus.zip" \
  "$SRC/gds_fdtd/examples/tech.yaml" \
  "$SRC/gds_fdtd/tests/tech_unified.yaml" \
  "$SRC/gds_fdtd/tests/tech_lumerical.yaml"

zip -j "$OUT/fuzz_dat_seed_corpus.zip" \
  "$SRC/gds_fdtd/tests/recorded/si_sin_escalator.dat"
