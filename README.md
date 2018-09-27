# AU: Search for "Gold"

This repo is a project to search for a PAC-style relation between
training data and accuracy for convolutional neural networks.

Named after Aurie Ginsberg


```
nvidia-docker run -d -it --name au1 -v `pwd`:/opt/au -v /:/outer_root -P --privileged tensorflow/tensorflow:1.10.1-gpu sleep infinity
docker exec -it -w /opt/au -e COLUMNS="`tput cols`" -e LINES="`tput lines`" au1 bash
```
