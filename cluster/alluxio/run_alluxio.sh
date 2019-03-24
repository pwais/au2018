#!/bin/bash
set -eux

AU_GLUSTER_URL=${AU_GLUSTER_URL:-"none"}
AU_GLUSTER_MOUNT_PATH=${AU_GLUSTER_MOUNT_PATH:-/media/au-gluster}
if [ "$AU_GLUSTER_URL" != "none" ]; then
  mkdir -p $AU_GLUSTER_MOUNT_PATH
  umount $AU_GLUSTER_MOUNT_PATH || true
  mount -t glusterfs $AU_GLUSTER_URL $AU_GLUSTER_MOUNT_PATH
  export ALLUXIO_UNDERFS_ADDRESS=${ALLUXIO_UNDERFS_ADDRESS:-$AU_GLUSTER_MOUNT_PATH}
  df -h
fi

# Alluxio needs these directories to be set up on the host
AU_ALLUXIO_MASTER_DIR=${AU_ALLUXIO_MASTER_DIR:-/opt/alluxio-master}
AU_ALLUXIO_TMP=${AU_ALLUXIO_TMP:-/tmp/alluxio}
mkdir -p $AU_ALLUXIO_MASTER_DIR/journal 
mkdir -p $AU_ALLUXIO_TMP/domain
chmod a+w $AU_ALLUXIO_TMP/domain
touch $AU_ALLUXIO_TMP/domain/d
chmod a+w $AU_ALLUXIO_TMP/domain/d

/opt/alluxio/integration/docker/entrypoint.sh $@
