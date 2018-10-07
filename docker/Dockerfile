FROM tensorflow/tensorflow:1.10.1-gpu

## Tensorflow
# Fix nvidia stubs and test tensorflow
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs
RUN \
  ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
  ldconfig && \
  python -c 'from tensorflow.python.client import device_lib; device_lib.list_local_devices()' 

## Dev tools
RUN \
  apt-get update && \
  apt-get install -y \
    curl \
    screen \
    sudo \
    vim \
    wget && \
  curl -LO https://github.com/BurntSushi/ripgrep/releases/download/0.10.0/ripgrep_0.10.0_amd64.deb && \
  dpkg -i ripgrep_0.10.0_amd64.deb

# Java
RUN \
  apt-get install -y default-jdk && \
  ls -lhat /usr/lib/jvm/java-8-openjdk-amd64 && \
  echo JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/environment
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64

# Python
#RUN pip install

COPY docker/.vimrc /root/.vimrc
COPY docker/.screenrc /root/.screenrc

## Alluxio
RUN \
  cd /tmp && \
  wget --progress=bar:force:noscroll http://downloads.alluxio.org/downloads/files/1.8.1/alluxio-1.8.1-hadoop-2.9-bin.tar.gz && \
  tar --checkpoint=1000 --checkpoint-action=dot -xzf alluxio-1.8.1-hadoop-2.9-bin.tar.gz -C /opt/ && \
  mv /opt/alluxio-1.8.1-hadoop-2.9 /opt/alluxio && \
  rm /tmp/alluxio-1.8.1-hadoop-2.9-bin.tar.gz

COPY docker/alluxio-site.properties /opt/alluxio/conf/alluxio-site.properties
