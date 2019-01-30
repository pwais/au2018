/opt/au/kubespray/inventory/default/artifacts# kubectl --kubeconfig admin.conf create configmap alluxio-config --from-env-file=/opt/au/cluster/alluxio/alluxio.properties
kubectl --kubeconfig admin.conf create -f /opt/au/cluster/alluxio/alluxio-master.yaml
kubectl --kubeconfig admin.conf logs -f pod/alluxio-master-0
kubectl --kubeconfig admin.conf create -f /opt/au/cluster/alluxio/alluxio-worker.yaml

http://au3:19999 to get to web ui

Set docker/alluxio-site.properties alluxio.underfs.address=/opt/alluxio-underfs if needed, or
set up alluxio.properties AU_GLUSTER_URL