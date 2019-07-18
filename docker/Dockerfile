FROM tensorflow/tensorflow:1.14.0-gpu-py3-jupyter
  # NB: bionic-based image

# We don't care for __pycache__ and .pyc files; sometimes VSCode doesn't clean
# up properly when deleting things and the cache gets stale.
ENV PYTHONDONTWRITEBYTECODE 1

# Use GCloud mirrors because they're faster, especially in gcloud
COPY docker/sources.list /etc/apt/sources.list

## Required for installing and testing things
RUN apt-get update
RUN \
  apt-get install -y \
  curl \
  git \
  python-dev \
  python-pip \
  python3-dev \
  wget



## Java 8.  NB: can't use Java 11 yet for Spark
RUN \
  apt-get install -y openjdk-8-jdk && \
  ls -lhat /usr/lib/jvm/java-8-openjdk-amd64 && \
  echo JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/environment
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64



## Tensorflow
# Fix nvidia stubs and test tensorflow.  Docker build will typically
# not run with a GPU, even if the machine has one, so this test
# will likely exercise the libcuda.so.1 stub library and show:
#   "failed call to cuInit: UNKNOWN ERROR (-1)"
#   "no NVIDIA GPU device is present: /dev/nvidia0 does not exist"
# which is a recoverable "error" on a non-gpu machine.  Without
# this stub in place, tensorflow-gpu will typically segfault on
# a machine with no GPUs
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64/stubs
RUN \
  ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
  ldconfig && \
  python3 -c 'from tensorflow.python.client import device_lib; device_lib.list_local_devices()' 



## Spark & Hadoop
ENV HADOOP_VERSION 3.1.1
ENV HADOOP_HOME /opt/hadoop
ENV HADOOP_CONF_DIR $HADOOP_HOME/etc/hadoop
ENV PATH $PATH:$HADOOP_HOME/bin
ENV LD_LIBRARY_PATH "$HADOOP_HOME/lib/native/:$LD_LIBRARY_PATH"
RUN curl -L --retry 3 \
  "http://mirrors.ibiblio.org/apache/hadoop/common/hadoop-$HADOOP_VERSION/hadoop-$HADOOP_VERSION.tar.gz" \
  | gunzip \
  | tar -x -C /opt/ \
 && mv /opt/hadoop-$HADOOP_VERSION $HADOOP_HOME \
 && rm -rf $HADOOP_HOME/share/doc

ENV SPARK_VERSION 2.4.3
ENV SPARK_PACKAGE spark-${SPARK_VERSION}-bin-without-hadoop
ENV SPARK_HOME /opt/spark
ENV SPARK_DIST_CLASSPATH "$HADOOP_HOME/etc/hadoop/*:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/tools/lib/*"
ENV PATH $PATH:${SPARK_HOME}/bin
RUN curl -L --retry 3 \
  "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz" \
  | gunzip \
  | tar x -C /opt/ \
 && mv /opt/$SPARK_PACKAGE $SPARK_HOME
COPY docker/spark-defaults.conf /opt/spark/conf/spark-defaults.conf



## Alluxio
RUN \
  apt-get install -y libfuse-dev && \
  cd /tmp && \
  wget --progress=bar:force:noscroll http://downloads.alluxio.org/downloads/files/1.8.1/alluxio-1.8.1-hadoop-2.9-bin.tar.gz && \
  tar --checkpoint=1000 --checkpoint-action=dot -xzf alluxio-1.8.1-hadoop-2.9-bin.tar.gz -C /opt/ && \
  mv /opt/alluxio-1.8.1-hadoop-2.9 /opt/alluxio && \
  rm /tmp/alluxio-1.8.1-hadoop-2.9-bin.tar.gz && \
  cp -v /opt/alluxio/client/alluxio-1.8.1-client.jar $SPARK_HOME/jars/ && \
  mkdir -p /opt/alluxio-cache && chmod 777 /opt/alluxio-cache && \
  mkdir -p /opt/alluxio-underfs && chmod 777 /opt/alluxio-underfs
#  mkdir -p /mnt/alluxio-ramdisk && mount -t ramfs -o size=1G ramfs /mnt/alluxio-ramdisk
COPY docker/alluxio-site.properties /opt/alluxio/conf/alluxio-site.properties



## Gluster
RUN \
  apt-get install -y software-properties-common && \
  add-apt-repository ppa:gluster/glusterfs-3.13 && \
  apt-get update && \
  apt-get install -y glusterfs-client && \
  glusterfs --version



## Dev tools
COPY docker/.vimrc /root/.vimrc
COPY docker/.screenrc /root/.screenrc
COPY cluster/devbox/start_ssh.sh /opt/au/cluster/devbox/start_ssh.sh
#COPY cluster/devbox/id_devbox_rsa.pub /opt/au/cluster/devbox/id_devbox_rsa.pub
RUN \
  apt-get update && \
  apt-get install -y \
    curl \
    dnsutils \
    git \
    gsutil \
    iputils-ping \
    net-tools \
    screen \
    ssh \
    sudo \
    tree \
    vim \
    wget && \
  pip3 install ipdb && \
  curl -LO https://github.com/BurntSushi/ripgrep/releases/download/0.10.0/ripgrep_0.10.0_amd64.deb && \
  dpkg -i ripgrep_0.10.0_amd64.deb

#ENV DOCKERVERSION 18.09.0-ce
#RUN curl -fsSLO https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKERVERSION}.tgz \
#  && tar xzvf docker-${DOCKERVERSION}.tgz --strip 1 \
#                 -C /usr/local/bin docker/docker \
#  && rm docker-${DOCKERVERSION}.tgz



## AU Python
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt
RUN python3 -c 'import imageio; imageio.plugins.ffmpeg.download()'

# Phantomjs for selenium, which we use for testing bokeh
RUN \
  cd /tmp && \
  wget https://bitbucket.org/ariya/phantomjs/downloads/phantomjs-2.1.1-linux-x86_64.tar.bz2 && \
  tar xvjf phantomjs-2.1.1-linux-x86_64.tar.bz2 && \
  cp phantomjs-2.1.1-linux-x86_64/bin/phantomjs /usr/bin/ && \
  rm -rf phantomjs-2.1.1-linux-x86_64 && \
  cd -


## TF Models
ENV PYTHONPATH $PYTHONPATH:/opt/au/external/tensorflow_models/research:/opt/au/external/tensorflow_models/research/slim:/opt/au/external/tensorflow_models/official:/opt/au/external/tensorflow_models/
RUN \
  DEBIAN_FRONTEND=noninteractive apt-get install -y protobuf-compiler python-pil python-lxml python-tk && \
  pip install Cython contextlib2 matplotlib && \
  cd /tmp && \
  git clone https://github.com/cocodataset/cocoapi.git && \
  cd cocoapi/PythonAPI && \
  python3 setup.py install



## TF CNN Vis
RUN pip3 install scipy h5py wget Pillow six scikit-image

## tsnet
RUN apt-get install -y libblas-dev libatlas-base-dev && pip3 install tsne

## Argoverse
RUN \
  apt-get install -y libsm6 libxext6 libxrender-dev && \
  mkdir -p /opt/argoverse && cd /opt/argoverse && \
  git clone https://github.com/argoai/argoverse-api.git && \
  git -C argoverse-api checkout 737793f2d4e0bf8f8b92a96b86ccb53be62c8ebc && \
  pip3 install -e argoverse-api && \
  cd -

## Kubespray
# See docker/kubespray_requirements.txt for notes
COPY docker/kubespray_requirements.txt /tmp/kubespray_requirements.txt
RUN pip3 install -r /tmp/kubespray_requirements.txt
ENV PATH $PATH:/opt/au/kubespray/inventory/default/artifacts/



## Gcloud
RUN \
  curl https://sdk.cloud.google.com | bash && \
  pip3 install -U crcmod 

# gcsfuse does NOT support read cache :P
# and s3fs is broken too it seems
#  export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
#  echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list && \
#  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - && \
#  apt-get update && \
#  apt-get install -y gcsfuse



## Smoke Test for AU
COPY cluster/alluxio /opt/au/cluster/alluxio
COPY aucli /opt/au/aucli
WORKDIR /opt/au
RUN \
  pip2 install findspark==1.3.0 && \
  SPARK_LOCAL_IP=127.0.0.1 SPARK_MASTER=local[4] /opt/au/aucli --test-spark 
