# AU: Search for "Gold"

This repo is a project to search for a PAC-style relation between
training data and accuracy for convolutional neural networks.

Named after Aurie Ginsberg



build a bridge.
 * try to repro the GAN paper thing wher we can sample from mnist layer and
    generate grad image via guided backprop
 * 


```
curl https://sdk.cloud.google.com | bash
```

```
gcloud auth login
gcloud config set project avian-augury-217704
gcloud components install docker-credential-gcr
gcloud auth configure-docker

--OR--
gcloud auth print-access-token | docker login -u oauth2accesstoken --password-stdin https://gcr.io

```


```
nvidia-docker run -d -it --name au1 -v `pwd`:/opt/au -v /:/outer_root --net=host -P --privileged tensorflow/tensorflow:1.10.1-gpu sleep infinity
docker exec -it -w /opt/au -e COLUMNS="`tput cols`" -e LINES="`tput lines`" au1 bash
```



```
gcloud ubuntu-minimal-1604-xenial-v20180814
cd external/kubespray
ansible-playbook -v --become -i kubespray/inventory/default/hosts.ini external/kubespray/cluster.yml


```




 * mnist, cifar 10?  bdd100k some vids, mscoco
 
 * mnist simple net
 * alexnet cifar 10
 * inception / maskrcnn on bdd100k, mscoco
 
 
 * write a tool to record activations.  take an activation and deconv to input image.
    what happens when we perturb activations a bit?
 
 * write a spark tool to take a model and collect all activations at scale
 * 




# TODO
 * docker image to serve as dev env.  try to set up GCR?
 * notebook or runbook for setting up a k8s cluster with gluster and alluxio and nvidia drivers
 * add some submodules
 * try to repro the deep segmentation thing
 * stand up mscoco and kitti and bdd tfrecords






http://ai-benchmark.com/ranking.html
https://github.com/PAIR-code/saliency
https://research.google.com/colaboratory/local-runtimes.html
https://openreview.net/forum?id=ryiAv2xAZ
https://web.eecs.umich.edu/~honglak/hl_publications.html
https://pair-code.github.io/saliency/
http://matthewalunbrown.com/mops/mops.html
https://github.com/clemenscorny/brisk/blob/master/LICENSE
https://arxiv.org/pdf/1312.6034.pdf
http://cmp.felk.cvut.cz/data/motorway/





do NOT use docker for mac due to host networking issue that has been known but unfixed since 2016 https://blog.bennycornelissen.nl/post/docker-for-mac-neat-fast-and-flawed/


https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb

https://github.com/conan7882/CNN-Visualization


hot stuff
exactly!  see edge-generated examples https://arxiv.org/pdf/1711.09325.pdf 
https://openreview.net/forum?id=ryiAv2xAZ
codes!!  https://github.com/alinlab/Confident_classifier 

also
https://github.com/ShiyuLiang/odin-pytorch
https://github.com/facebookresearch/odin <-- hmm CC license

can we use LSH to sample from feature map latent space?  scooped!
https://arxiv.org/pdf/1802.07444.pdf
 

https://arxiv.org/pdf/1610.02136.pdf ... better! out of distribution !!! nips 18   https://arxiv.org/pdf/1807.03888.pdf   and detecting novel https://arxiv.org/pdf/1804.00722.pdf   https://web.eecs.umich.edu/~honglak/hl_publications.html   https://arxiv.org/pdf/1711.09325.pdf
https://arxiv.org/pdf/1705.08664.pdf  could be very good metric, seem to use gaussian input to generate images tho using a network?

daylen https://arxiv.org/pdf/1701.02362.pdf 

