# Based upon the pyspark k8s Dockerfile:
# https://github.com/apache/spark/blob/0889fbaf959e25ebb79e691692a02a93962727d0/resource-managers/kubernetes/docker/src/main/dockerfiles/spark/bindings/python/Dockerfile

ARG base_img
ARG spark_uid=185

FROM $base_img

WORKDIR /opt/spark/work-dir
RUN chmod g+w /opt/spark/work-dir

COPY entrypoint.sh /opt/
RUN chmod +x /opt/entrypoint.sh

ENTRYPOINT [ "/opt/entrypoint.sh" ]
USER ${spark_uid}