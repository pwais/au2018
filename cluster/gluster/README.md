sudo su -
mkdir -p /etc/glusterfs /var/lib/glusterd /var/log/glusterfs
mkdir -p /media/seagates/brick
sysctl -w kernel.core_pattern=/var/log/core/core_%e.%p

gluster peer status
gluster volume create aubrick1 au2:/media/brick/brick au3:/media/brick/brick
gluster volume start aubrick1
gluster volume info


sudo mkdir /media/testaubrick
sudo mount -t glusterfs au2:/aubrick1 /media/testaubrick
df -h