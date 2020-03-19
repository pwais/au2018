#!/bin/bash
# Based upon https://docs.docker.com/engine/examples/running_ssh_service/

set -eux

mkdir -p /var/run/sshd
mkdir -p /root/.ssh
cat /opt/au/cluster/devbox/id_devbox_rsa.pub >> /root/.ssh/authorized_keys

while true; do /usr/sbin/sshd -p 30022 -D -d || true ; done
