# ClusterFuzzLite build image (Python / atheris).
FROM gcr.io/oss-fuzz-base/base-builder-python

COPY . $SRC/gds_fdtd
WORKDIR $SRC/gds_fdtd
COPY .clusterfuzzlite/build.sh $SRC/build.sh
