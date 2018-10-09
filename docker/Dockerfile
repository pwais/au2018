FROM tensorflow/tensorflow:1.10.1-gpu


## Java
RUN \
  apt-get install -y default-jdk && \
  ls -lhat /usr/lib/jvm/java-8-openjdk-amd64 && \
  echo JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/environment
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64


## Tensorflow
# Fix nvidia stubs and test tensorflow
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64/stubs
RUN \
  ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
  ldconfig && \
  python -c 'from tensorflow.python.client import device_lib; device_lib.list_local_devices()' 

COPY docker/sources.list /etc/apt/sources.list


## Dev tools
RUN \
  apt-get update && \
  apt-get install -y \
    curl \
    git \
    screen \
    ssh \
    sudo \
    vim \
    wget && \
  curl -LO https://github.com/BurntSushi/ripgrep/releases/download/0.10.0/ripgrep_0.10.0_amd64.deb && \
  dpkg -i ripgrep_0.10.0_amd64.deb

COPY docker/.vimrc /root/.vimrc
COPY docker/.screenrc /root/.screenrc



# Python
#RUN pip install



## Spark
#RUN \
#  wget 

## Alluxio
RUN \
  apt-get install -y libfuse-dev && \
  cd /tmp && \
  wget --progress=bar:force:noscroll http://downloads.alluxio.org/downloads/files/1.8.1/alluxio-1.8.1-hadoop-2.9-bin.tar.gz && \
  tar --checkpoint=1000 --checkpoint-action=dot -xzf alluxio-1.8.1-hadoop-2.9-bin.tar.gz -C /opt/ && \
  mv /opt/alluxio-1.8.1-hadoop-2.9 /opt/alluxio && \
  rm /tmp/alluxio-1.8.1-hadoop-2.9-bin.tar.gz && \
  mkdir -p /opt/alluxio-cache && chmod 777 /opt/alluxio-cache && \
  mkdir -p /opt/alluxio-underfs && chmod 777 /opt/alluxio-underfs

COPY docker/alluxio-site.properties /opt/alluxio/conf/alluxio-site.properties


## TF Models
ENV PYTHONPATH $PYTHONPATH:/opt/au/external/tensorflow_models/research:/opt/au/external/tensorflow_models/research/slim
RUN \
  apt-get install -y protobuf-compiler python-pil python-lxml python-tk && \
  pip install Cython contextlib2 matplotlib && \
  cd /tmp && \
  git clone https://github.com/cocodataset/cocoapi.git && \
  cd cocoapi/PythonAPI && \
  python setup.py install


## TF CNN Vis
RUN pip install scipy h5py wget Pillow six scikit-image


## Kubespray
COPY external/kubespray/requirements.txt /tmp/kubespray_requirements.txt
RUN pip install -r /tmp/kubespray_requirements.txt

## Gcloud
RUN \
  curl https://sdk.cloud.google.com | bash && \
  pip install -U crcmod

