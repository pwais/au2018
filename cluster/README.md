# Cluster

This directory contains utilities for building a private cluster.

Requirements:
 1. Docker registry (e.g. Docker Hub)
 2. Alluxio-compatible cloud storage (e.g. S3 or GS)
 3. Ubuntu 16.04 (at the time of writing, `kubespray` has issues with Bionic).
 4. Run [sudo setup.sh](setup.sh) script to install basic packages and nvidia
      drivers (if possible).  You will need to reboot for Nvidia driver
      changes to take effect.
 5. Passwordless `ssh` and `sudo` for cluster deployment and operation.

We've tested these utilites with bare metal machines as well as GCE intances
(using Google's own Ubuntu 16.04 image -- `ubuntu-minimal-1604-xenial-v20180814`).

## GCloud Setup - Storage

GCloud currently has a free trial promotion that includes USD$300 of
credits.  We'll use this trial primarily for cloud storage, but the
GCE GPU offerings are a useful compute resource.

1. Create an account
2. Add your ssh key
3. Create a bucket, recommend with regional storage (lower price)
      in US Central (currently the zone with best price per GPU).
      Use the "enable interoperability" feature to 
   *Put your GCS keys in [my.env](.gitignore#L2); see also
   [my.env.example](my.env.example)*

## Deploying Kubernetes (k8s) - GCloud or Bare Metal

### Why?

We use `kubespray` and k8s for the following reasons:
 * `kubespray` makes k8s deployment extremely easy with Ubuntu; one can
        add and remove nodes without major issue.
 * `kubespray` is cloud-agnostic.
 * k8s can serve as a [Spark cluster manager](https://spark.apache.org/docs/latest/running-on-kubernetes.html)

### How?

 1. Create and edit [hosts.ini](kubespray/inventory/default/hosts.ini) to suit
        your needs.  See [the example](kubespray/inventory/default/hosts.ini.example).
 2. Run setup:
        ```
        cd external/kubespray
        ansible-playbook -v --become -i kubespray/inventory/default/hosts.ini external/kubespray/cluster.yml
        ```