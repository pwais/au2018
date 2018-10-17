FROM tensorflow/tensorflow:1.10.1-gpu

COPY docker/sources.list /etc/apt/sources.list

RUN apt-get update

## Java
RUN \
  apt-get install -y default-jdk && \
  ls -lhat /usr/lib/jvm/java-8-openjdk-amd64 && \
  echo JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 >> /etc/environment
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64


## Tensorflow
# Fix nvidia stubs and test tensorflow
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/lib:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64/stubs
RUN \
  ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
  ldconfig && \
  python -c 'from tensorflow.python.client import device_lib; device_lib.list_local_devices()' 

## Dev tools
COPY docker/.vimrc /root/.vimrc
COPY docker/.screenrc /root/.screenrc
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

## Bazel -- might not need this ...
RUN \
  apt-get install -y bash-completion && \
  wget https://github.com/bazelbuild/bazel/releases/download/0.16.1/bazel_0.16.1-linux-x86_64.deb && \
  dpkg -i bazel_0.16.1-linux-x86_64.deb 
  

## Python
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt


## Spark
ENV HADOOP_VERSION 2.9.1
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

ENV SPARK_VERSION 2.3.2
ENV SPARK_PACKAGE spark-${SPARK_VERSION}-bin-without-hadoop
ENV SPARK_HOME /opt/spark
ENV SPARK_DIST_CLASSPATH "$HADOOP_HOME/etc/hadoop/*:$HADOOP_HOME/share/hadoop/common/lib/*:$HADOOP_HOME/share/hadoop/common/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/hdfs/lib/*:$HADOOP_HOME/share/hadoop/hdfs/*:$HADOOP_HOME/share/hadoop/yarn/lib/*:$HADOOP_HOME/share/hadoop/yarn/*:$HADOOP_HOME/share/hadoop/mapreduce/lib/*:$HADOOP_HOME/share/hadoop/mapreduce/*:$HADOOP_HOME/share/hadoop/tools/lib/*"
ENV PATH $PATH:${SPARK_HOME}/bin
RUN curl -L --retry 3 \
  "https://www.apache.org/dyn/mirrors/mirrors.cgi?action=download&filename=spark/spark-${SPARK_VERSION}/${SPARK_PACKAGE}.tgz" \
  | gunzip \
  | tar x -C /opt/ \
 && mv /opt/$SPARK_PACKAGE $SPARK_HOME \
 && pip install findspark

## Alluxio
RUN \
  apt-get install -y libfuse-dev && \
  cd /tmp && \
  wget --progress=bar:force:noscroll http://downloads.alluxio.org/downloads/files/1.8.1/alluxio-1.8.1-hadoop-2.9-bin.tar.gz && \
  tar --checkpoint=1000 --checkpoint-action=dot -xzf alluxio-1.8.1-hadoop-2.9-bin.tar.gz -C /opt/ && \
  mv /opt/alluxio-1.8.1-hadoop-2.9 /opt/alluxio && \
  rm /tmp/alluxio-1.8.1-hadoop-2.9-bin.tar.gz && \
  mkdir -p /opt/alluxio-cache && chmod 777 /opt/alluxio-cache && \
  mkdir -p /opt/alluxio-underfs && chmod 777 /opt/alluxio-underfs && \
  mkdir -p /mnt/alluxio-ramdisk && mount -t ramfs -o size=1G ramfs /mnt/alluxio-ramdisk && 
COPY docker/alluxio-site.properties /opt/alluxio/conf/alluxio-site.properties

## TF Models
ENV PYTHONPATH $PYTHONPATH:/opt/au/external/tensorflow_models/research:/opt/au/external/tensorflow_models/research/slim:/opt/au/external/tensorflow_models/official:/opt/au/external/tensorflow_models/
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

# gcsfuse does NOT support read cache :P
# and s3fs is broken too it seems
#  export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s` && \
#  echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list && \
#  curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add - && \
#  apt-get update && \
#  apt-get install -y gcsfuse

