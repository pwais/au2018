#!/bin/bash

set -eux 

# Install basics for 16.04 image: http://releases.ubuntu.com/16.04/ubuntu-16.04.5-desktop-amd64.iso

apt-get update
apt-get install -y \
  curl \
  git \
  openssh-server \
  screen \
  ssh \
  vim \
  wget
service ssh restart
ubuntu-drivers autoinstall

## Docker
#curl -fsSL https://get.docker.com | bash
# kubespray wants this version of docker
service docker stop || true
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"
apt-get update
apt-get install -y docker-ce=18.06.1~ce~3-0~ubuntu
usermod -aG docker au

## nvidia-docker
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
apt-get update
apt-get install -y nvidia-docker2
pkill -SIGHUP dockerd

# Passwordless sudo
echo "au ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

echo "Reboot and test with $ nvidia-docker run -it --rm nvidia/cuda:9.0-base nvidia-smi"