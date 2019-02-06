# `au` Cluster

This directory contains utilities for building a private cluster running
Kubernetes (via Kubespray), Spark, and more.  The cluster design has the
following in mind:
 * The cluster uses Spark for running jobs, and Kubernetes as a Spark master / 
      container orchestrator.  Some jobs (e.g. a deep learning training job)
      may take exclusive access of an entire machine as it runs.
      `au` facilitates exclusive access.
 * The cluster persists long-lived data to a distributed filesystem (e.g. 
      Gluster or S3), and the worker nodes leverage caching to make
      read access efficient.  In particular, the cluster can leverage
      Alluxio to provide a local-SSD-based cache on each worker.
 * The cluster has only light security features.  Network partitioning is
      a primary defense.  Run the cluster behind a NAT or in a VPC.
 * The cluster is not designed to be multi-user but does not preclude
      cooperative multi-user jobs scheduling.  (E.g. just put an Executor
      limit on your Spark jobs).

Requirements:
 1. Ideally at least 2 and as many as O(1000) machines, either bare metal, 
      [GCE Compute Instances](https://cloud.google.com/free/), AWS EC2
      instances, etc.  Machines should have local SSD disks for caching.
 2. Ubuntu 16.04 (at the time of writing, `kubespray` has issues with Bionic).
 3. Passwordless `ssh` cluster deployment.
 4. (Recommended) a large persistent filesystem, e.g. AWS S3 or Google Storage,
      or disks for use in a Gluster-based fileystem ([see below](#Gluster)).
 5. Read access to a Docker registry (e.g. Docker Hub; you can use the
      public `au` images).
 6. Software dependencies?  If your machines have GPUs, either run
      `sudo cluster/setup_16.04.sh` to install basic packages and Nvidia
      drivers, or install the software listed in that script.  You will need to
      reboot for Nvidia driver changes to take effect.

      If your machines already have Nvidia drivers and `nvidia` is the
      `default-runtime` for Docker, then you don't need to run our setup
      script.

      If your machines do not have GPUs, then Ansible / Kubespray will install
      needed dependencies; all you need is `python` and passwordless `ssh` 
      access.

These utilites have been tested with bare metal machines as well as GCE intances
(using Google's own Ubuntu 16.04 image -- `ubuntu-minimal-1604-xenial-v20180814`).


## Part 1: Deploy Kubernetes (k8s) - GCloud or Bare Metal

### Why?

We use `kubespray` and k8s for the following reasons:
 * `kubespray` makes k8s deployment easy with Ubuntu; one can add and remove
        nodes without major issues, and ansible is fairly effective at
        handling machine state.
 * `kubespray` is cloud-agnostic.
 * k8s can serve as a [Spark cluster manager](https://spark.apache.org/docs/latest/running-on-kubernetes.html)

### How?

 1. Run `./aucli --shell` to drop into a dockerized shell.
 2. Use `./aucli --kube-init` to step through cluster configuation set-up.
       You'll need a cluster ssh key and a [hosts.ini](kubespray/inventory/default/hosts.ini.example)
       file to spec the cluster.
 3. Use `./aucli --kube-up` to bring up the cluster via kubespray.  Kubespray
       may fail the first time or two, or you may have other bugs.  `aucli`
       will print out the commands it runs, so try reviewing stdout in order
       to debug.
 4. The inventory config in the `au` repo includes `kubectl_localhost: true`
       and `kubeconfig_localhost: true` to allow local k8s access.  To test
       your cluster as well as get the path to `kubectl`, use:
              ```./aucli --kube-test```
       inside the shell and look for "Path to kubectl".

#### Useful Links
 * https://github.com/kubernetes-sigs/kubespray/blob/master/docs/getting-started.md
 * To debug ansible, try `ansible-playbook -vvv` (more verbosity)


## Part 2: Stand Up Persistant Storage

## Single Local Storage Device

If you want to persist your data to only one disk, you will likely want to
mount that disk on each worker node.  Try using NFS or SSHFS / Fuse.  (We 
wont' address this use case in detail).

## Gluster

If your machines have a relatively fast connection to privately-managed 
storage (e.g. a bunch of USB back-up drives attached to your boxen), then 
we recommend you use Gluster to create a distributed filesystem spanning
this storage.  In practice, Gluster can scale effectively to tens of 
petabytes of disk supporting OLAP workloads.  Gluster administration is not
easy, but see the [Gluster Guide](gluster/) to set up a simple Gluster cluster
that works well in practice.

## GCloud Setup - Storage

GCloud currently has a free trial promotion that includes USD$300 of
credits.  We'll use this trial primarily for cloud storage, but the
GCE GPU offerings are a useful compute resource.

1. Create an account
2. Add your ssh key
3. Create a bucket, recommend with regional storage (lower price)
      in US Central (currently the zone with best price per GPU).
      Use the "enable interoperability" feature to 
   *Put your GCS keys in [my.env](../.gitignore#L2); see also
   [my.env.example](../my.env.example)*


## Part 3: Stand Up Alluxio (Caching)

