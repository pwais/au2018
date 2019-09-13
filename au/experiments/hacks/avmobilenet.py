
from au.spark import Spark
from au.spark import spark_df_to_tf_dataset

from au import conf
from au import util

import tensorflow as tf
import numpy as np

from au.fixtures.datasets.auargoverse import AV_OBJ_CLASS_TO_COARSE
AV_OBJ_CLASS_NAME_TO_ID = dict(
  (cname, i + 1)
  for i, cname in enumerate(sorted(AV_OBJ_CLASS_TO_COARSE.keys())))
AV_OBJ_CLASS_NAME_TO_ID['background'] = 0






class model_fn_vae_hybrid(object):

  def __init__(self, au_params):
    self.params = au_params

    # from keras import backend as K
    # K.set_learning_phase(0)
    tf.keras.backend.set_learning_phase(0)
    with util.tf_cpu_session() as sess:
      import keras.applications.resnet50 as resnet50
      resnet50 = resnet50.ResNet50(weights='imagenet', include_top=False)
      graph = sess.graph


      # # fix batch norm nodes
      # for node in sess.graph_def.node:
      #   if node.op == 'RefEnter':
      #     node.op = 'Enter'
      #     for index in range(len(node.input)):
      #       if 'moving_' in node.input[index]:
      #         node.input[index] = node.input[index] + '/read'
      #   if node.op == 'RefSwitch':
      #     node.op = 'Switch'
      #     for index in range(len(node.input)):
      #       if 'moving_' in node.input[index]:
      #         node.input[index] = node.input[index] + '/read'
      #   elif node.op == 'AssignSub':
      #     node.op = 'Sub'
      #     if 'use_locking' in node.attr: del node.attr['use_locking']
      #   elif node.op == 'AssignAdd':
      #     node.op = 'Add'
      #     if 'use_locking' in node.attr: del node.attr['use_locking']
      
      # Clear devices
      graph_def = graph.as_graph_def()
      for node in graph_def.node:
        node.device = ""

      self.resnet_50_in_layers = ['input_1']
      self.resnet_50_out_layers = ['activation/Relu'] + [ # ~~~~~['conv1/BiasAdd'] + [
        'activation_%s/Relu' % a
        for a in (2, 5, 10, 20, 30, 48)
      ]
      layers = self.resnet_50_in_layers + self.resnet_50_out_layers
      frozen_graph = tf.graph_util.convert_variables_to_constants(
          sess,
          graph_def,
          layers)
      self.resnet50_graph = tf.graph_util.remove_training_nodes(frozen_graph)
      print('saved graph')

    tf.keras.backend.clear_session()
    # K.clear_session()
  
  def __call__(self, features, labels, mode, params):

    FEATURES_SHAPE = [None, 170, 170, 3]
    N_CLASSES = len(AV_OBJ_CLASS_NAME_TO_ID)

    # Downweight background class
    class_weights = np.ones(len(AV_OBJ_CLASS_NAME_TO_ID))
    BKG_CLASS_IDX = AV_OBJ_CLASS_NAME_TO_ID['background']
    class_weights[BKG_CLASS_IDX] *= .1
    class_weights[AV_OBJ_CLASS_NAME_TO_ID["VEHICLE"]] *= .5
    class_weights /= class_weights.sum()
    cweights = tf.gather(class_weights, labels)

    features.set_shape(FEATURES_SHAPE) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    labels.set_shape([None,])

    obs_str_tensor = util.ThruputObserver.monitoring_tensor('features', features)

    # Mobilenet requires normalization
    features_norm = tf.cast(features, tf.float32) / 128. - 1

    with tf.name_scope('input'):
      util.tf_variable_summaries(tf.cast(labels, tf.float32), prefix='labels')
      util.tf_variable_summaries(features_norm, prefix='features_norm')

    ### Set up Mobilenet
    from nets.mobilenet import mobilenet_v2
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    scope = mobilenet_v2.training_scope(is_training=is_training)
    with tf.contrib.slim.arg_scope(scope):
      # image (?, 170, 170, 3) -> embedding (?, 6, 6, 1280)
      _, endpoints = mobilenet_v2.mobilenet(
                              features_norm,
                              num_classes=N_CLASSES,
                              base_only=True,
                              is_training=is_training,
                              depth_multiplier=self.params.DEPTH_MULTIPLIER,
                              finegrain_classification_mode=self.params.FINE)
      
      if is_training:
        # Per authors: Restore using exponential moving average since it produces
        # (1.5-2%) higher accuracy
        ema = tf.train.ExponentialMovingAverage(0.999)
        vs = ema.variables_to_restore()

        import os
        util.download(
          self.params.CHECKPOINT_TARBALL_URI,
          self.params.MODEL_BASEDIR)
        checkpoint = os.path.join(
          self.params.MODEL_BASEDIR,
          self.params.CHECKPOINT + '.ckpt')
        
        tf.train.init_from_checkpoint(
          checkpoint, {'MobilenetV2/' : 'MobilenetV2/'})

      # layers_to_pick = ['layer_%s' % (i + 1) for i in range(19)]
      layers_to_pick = ['layer_%s' % i for i in (2, 5, 10, 15, 19)]
      mobilenet_layers = dict((l, endpoints[l]) for l in layers_to_pick)
      for l_name, l in mobilenet_layers.items():
        util.tf_variable_summaries(l, prefix=l_name)

      # embedding = endpoints['layer_19']
  
    ### Set up the VAE model
    class VAELayer(object):
      pass
    def create_vae_layer(layer_in, name, prev=None):
      with tf.variable_scope(name):
        Z_D = 4
        ENCODER_LAYERS = [128, 64]
        DECODER_LAYERS = [64, 128]
        
        l = tf.keras.layers
        
        vae_layer = VAELayer()
        vae_layer.name = name

        layer_in_flat = tf.contrib.layers.flatten(layer_in)
        util.tf_variable_summaries(layer_in)
        
        ## Encode
        ## x -> z
        encode = tf.keras.Sequential([
          l.Dense(n_hidden, activation=tf.nn.relu6, name='encode_%s' % i)
            for i, n_hidden in enumerate(ENCODER_LAYERS)
          ])
        encoded = encode(layer_in_flat, training=is_training)
        if prev is not None:
          encoded += prev.encoded
        util.tf_variable_summaries(encoded)
        vae_layer.encoded = encoded

        ## Latent Variables
        z_mu_layer = l.Dense(Z_D, name='z_mu_layer')
        z_mu = z_mu_layer(vae_layer.encoded)
        util.tf_variable_summaries(z_mu)
        vae_layer.z_mu = z_mu

        z_var_layer = l.Dense(
          Z_D, activation=tf.nn.softplus, name='z_var_layer')
        z_var = z_var_layer(vae_layer.encoded)
        util.tf_variable_summaries(z_var)
        vae_layer.z_var = z_var

        noise = tf.keras.backend.random_normal(
                  shape=tf.shape(z_var),
                  mean=0,
                  stddev=1,
                  dtype=tf.float32,
                  seed=util.stable_hash(name))
        z = z_mu + z_var * noise
        util.tf_variable_summaries(z)
        vae_layer.z = z

        ## Latent Loss: KL divergence between Z and N(0, 1)
        latent_loss = tf.reduce_mean(
              -0.5 * tf.reduce_sum(
                        1 + tf.log(z_var) - z_var - tf.square(z_mu),
                      axis=1))
        tf.summary.scalar(name + '/latent_loss', latent_loss)
        vae_layer.latent_loss = latent_loss

        ## Decode
        ## z -> y
        decode = tf.keras.Sequential([
          l.Dense(n_hidden, activation=tf.nn.relu6, name='decode_%s' % i)
          for i, n_hidden in enumerate(DECODER_LAYERS)
        ])
        y = decode(z, training=is_training)
        util.tf_variable_summaries(y)
        vae_layer.y = y

        return vae_layer

    vae_layers = []
    prev = None
    for l_name, layer in mobilenet_layers.items():
      layer = layer / 3 - 1
        # Mobilenet uses tf.nn.relu6
      vae_layer = create_vae_layer(layer, l_name, prev=prev)
      vae_layers.append(vae_layer)
      prev = vae_layer

      # tf.summary.histogram(l_name + '_sss', 
      #     tf.log(vae_layer.z_var) - vae_layer.z_var - tf.square(vae_layer.z_mu))
    
    total_latent_loss = tf.constant(0.0)
    for vl in vae_layers:
      total_latent_loss += vl.latent_loss
    tf.summary.scalar('total_latent_loss', total_latent_loss)





            # ### VAE Input
            # mn_layers = [
            #   tf.contrib.layers.flatten(mobilenet_layers[l])
            #   for l in sorted(mobilenet_layers.keys())
            # ]
            # x = tf.concat(mn_layers, axis=-1)
            # x = x / 3 - 1
            #   # Mobilenet uses tf.nn.relu6
            # # x = tf.identity(embedding / 3 - 1, name='x')
            # util.tf_variable_summaries(x)

            # # ## Encode
            # # ## x -> z = N(z_mu, z_sigma)
            # l = tf.keras.layers
            # encode = tf.keras.Sequential([
            #   l.Dense(n_hidden, activation=None, name='encode_%s' % i)
            #   for i, n_hidden in enumerate(ENCODER_LAYERS)
            # ])
            # x_flat = tf.contrib.layers.flatten(x)
            # encoded = encode(x_flat, training=is_training)

            # z_mu_layer = l.Dense(Z_D, name='z_mu_layer')
            # z_mu = z_mu_layer(encoded)

            # z_var_layer = l.Dense(Z_D, activation=tf.nn.softplus, name='z_var_layer')
            # z_var = z_var_layer(encoded)

            # util.tf_variable_summaries(z_mu)
            # util.tf_variable_summaries(z_var)

            # noise = tf.keras.backend.random_normal(
            #           shape=tf.shape(z_var),
            #           mean=0,
            #           stddev=1,
            #           dtype=tf.float32)
            # z = z_mu + z_var * noise
            # # z = tf.random_normal(
            # #         tf.shape(z_mu),
            # #         mean=z_mu, 
            # #         stddev=tf.sqrt(z_var),
            # #         seed=1337,
            # #         name='z')
            # util.tf_variable_summaries(z)

            # ## Latent Loss: KL divergence between Z and N(0, 1)
            # # latent_loss = tf.reduce_mean(
            # #   -0.5 * tf.reduce_sum(
            # #     1. + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq),
            # #     axis=1))
            # # latent_loss = -0.5 * tf.reduce_sum(
            # #           1. + z_log_sigma_sq - tf.square(z_mu) - tf.exp(z_log_sigma_sq))
            # latent_loss = tf.reduce_mean(
            #       -0.5 * tf.reduce_sum(
            #             1 + tf.log(z_var) - z_var - tf.square(z_mu), axis=1))
            # # latent_loss = tf.reduce_mean(
            # #     -0.5 * tf.reduce_sum(
            # #         tf.log(2 * np.pi) +
            # #           tf.log(z_var) +
            # #           tf.square(z - z_mu) / z_var,
            # #         axis=1))
            # tf.summary.scalar('latent_loss', latent_loss)

            # ## Decode
            # ## z -> y
            # decode = tf.keras.Sequential([
            #   l.Dense(n_hidden, activation=None, name='decode_%s' % i)
            #   for i, n_hidden in enumerate(DECODER_LAYERS)
            # ])
            # y = decode(z, training=is_training)
            # util.tf_variable_summaries(y)
    
    ## Class Prediction Head
    ## Y -> class'; loss(class, class')
    # TODO: consider dedicating latent vars to this head as in
    # http://people.csail.mit.edu/rosman/papers/iros-2018-variational.pdf

    ys = [vl.y for vl in vae_layers]
    Y = tf.concat(ys, axis=-1)

    predict_layer = tf.keras.layers.Dense(N_CLASSES, activation=None)
    logits = predict_layer(Y)
    labels_pred = tf.nn.softmax(logits)

    util.tf_variable_summaries(logits)
    multiclass_loss = tf.losses.sparse_softmax_cross_entropy(
                        labels=labels,
                        logits=logits,
                        weights=cweights)
    tf.summary.scalar('multiclass_loss', multiclass_loss)

    preds = tf.argmax(labels_pred, axis=-1)
    tf.summary.histogram('logits', logits)
    tf.summary.histogram('preds', preds)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)
    tf.summary.scalar('accuracy', accuracy[1])

    

    #   # # TODO: keras.losses.binary_crossentropy ? TODO try L1 loss?
    #   # # http://people.csail.mit.edu/rosman/papers/iros-2018-variational.pdf
    #   # loss = tf.reduce_mean(
    #   #   -tf.reduce_sum(
    #   #     y * tf.log(y_ + EPS) + (1 - y) * tf.log(1. - y_ + EPS), axis=1))
    #   # tf.debugging.check_numerics(y, 'y_nan')
    #   # tf.debugging.check_numerics(y_, 'y_target_nan')
    #   # tf.debugging.check_numerics(loss, 'loss_nan')

    # ## Reconstruction Head: Image
    # ## y -> image'; loss(image, image')
    # image = features_norm
    # with tf.name_scope('decode_image'):
    #   n_y = int(y.shape[-1])

    #   # Like StyleGAN we start from a learnable constant and let `y` steer
    #   # the conv filters applied to this constant.
    #   # base = tf.get_variable(
    #   #           'base',
    #   #           shape=[1, 4, 4, n_y],
    #   #           initializer=tf.initializers.ones())
    #   base = l.Dense(4 * 4 * 256, activation=None)(y)
    #   # base_batched = tf.tile(base, [tf.shape(image)[0], 1, 1, 1])
    #   base_batched = tf.reshape(base, [-1, 4, 4, 256])

    #   image_h, image_w = (int(image.shape[1]), int(image.shape[2]))
    #   filters = [ 256, 128,  64,  64,  64]
    #   # filters = [n_y,   64,  64,  64,  64]
    #   scales =  [0.05, 0.1, 0.2, 0.4, 0.8]
    #   assert len(filters) == len(scales)
    #   x = base_batched
    #   for s, f in zip(scales, filters):
    #     with tf.name_scope('scale_%s' % s):
    #       x = l.Conv2D(
    #             filters=f, kernel_size=3, activation=tf.nn.relu6,
    #             strides=1, padding='same', input_shape=x.shape[1:])(x)

    #       # # AdaIN: Instance Norm, convert `current` to z-scores
    #       # x -= tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    #       # x_ss = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
    #       # x *= (tf.rsqrt(x_ss) + 1e-8)
          
    #       # # AdaIn: Adaptive Scaling (with learned scaling factors ...)...
    #       # # Based upon https://github.com/NVlabs/stylegan/blob/f3a044621e2ab802d40940c16cc86042ae87e100/training/networks_stylegan.py#L259
    #       # x_scale = l.Dense(f, activation=tf.nn.relu6, name='x_scale')(y)
    #       # x_bias = l.Dense(f, activation=tf.nn.relu6, name='x_bias')(y)
    #       # util.tf_variable_summaries(x_scale)
    #       # util.tf_variable_summaries(x_bias)

    #       # x_scale = tf.reshape(x_scale, [-1, 1, 1, x_scale.shape[-1]])
    #       # x_bias = tf.reshape(x_bias, [-1, 1, 1, x_bias.shape[-1]])
    #       # x = (x_scale + 1) * x + x_bias
    #       x = tf.nn.relu6(x)
    #       util.tf_variable_summaries(x)

    #       # Use a bilinear resize instead of a deconv; it's visibly much
    #       # smoother and the upscaling is much easier to control.
    #       h_out, w_out = int(s * image_h), int(s * image_w)
    #       x = tf.image.resize_images(
    #                   x, [h_out, w_out],
    #                   method=tf.image.ResizeMethod.BILINEAR,
    #                   align_corners=True)
    #   image_hat_decoded = x
      

    #   # decode_image = tf.keras.Sequential([
    #   #   l.Convolution2DTranspose(
    #   #     filters=f, kernel_size=3, activation=tf.nn.relu6,
    #   #     strides=2, padding='same')
    #   #   for f in filters
    #   # ])
      
    #   # # For GANs, it seems people just randomly reshape noise into
    #   # # some dimensions that are plausibly deconv-able.  Here, we try to
    #   # # amplify `y` such that every conv kernel can sample *every* value
    #   # # of `y`; something simple like tf.reshape(y) will arbitrarily
    #   # # restrict the receptive field of each kernel.
    #   # tile_h, tile_w = (3, 3)
    #   # row = tf.stack([y] * tile_w, axis=1)
    #   # y_expanded = tf.stack([row] * tile_h, axis=2)

    #   # decoded_image_base = decode_image(y_expanded, training=is_training)
    
    #   # Finally, apply upsample trick for perfect fit
    #   # https://github.com/SimonKohl/probabilistic_unet/blob/master/model/probabilistic_unet.py#L60
    #   image_c = int(image.shape[-1])
    #   upsampled = tf.image.resize_images(
    #                   image_hat_decoded, [image_h, image_w],
    #                   method=tf.image.ResizeMethod.BILINEAR,
    #                   align_corners=True)
    #   image_hat_layer = tf.keras.layers.Conv2D(
    #                         filters=image_c, kernel_size=5,
    #                         activation='tanh',
    #                           # Need to be in [-1, 1] to match image domain
    #                         padding='same')
    #   image_hat = image_hat_layer(upsampled)
    #   util.tf_variable_summaries(image_hat)
    
    # tf.summary.image(
    #   'reconstruct_image', image, max_outputs=10)
    # tf.summary.image(
    #   'reconstruct_image_hat', image_hat, max_outputs=10)
    
    # # Base loss: try to reconstruct the input image.  
    # base_recon_loss = tf.losses.absolute_difference(
    #                       tf.contrib.layers.flatten(image), 
    #                       tf.contrib.layers.flatten(image_hat),
    #                       weights=tf.expand_dims(cweights, axis=-1))
    # tf.summary.scalar('base_recon_loss', base_recon_loss)
    # recon_image_loss = 100.0 * base_recon_loss
    #   # But this is not enough for convergence, though, so we add ...

    # # ... Perceptual Loss
    # with tf.name_scope('perceptual_loss'):
    #   PL_SCALES = (1.0, )#0.5,)# 0.1)
    #   for scale in PL_SCALES:
    #     with tf.name_scope('scale_%s' % scale):
    #       def downsize(im):
    #         if scale == 1:
    #           return im
    #         h_out, w_out = int(scale * image_h), int(scale * image_w)
    #         return tf.image.resize_images(
    #                           im, [h_out, w_out],
    #                           method=tf.image.ResizeMethod.BICUBIC,
    #                           align_corners=True)
          
    #       image_s = downsize(image)
    #       image_hat_s = downsize(image_hat)

    #       def preprocess(x):
    #         x = 255 * (x + 1)
    #         x = tf.image.resize_images(
    #                         x, (224, 224),
    #                         method=tf.image.ResizeMethod.BICUBIC,
    #                         align_corners=True)
    #         from keras.applications import resnet50
    #         x = resnet50.preprocess_input(x)
    #         return x
    #       image_processed = preprocess(image_s)
    #       image_hat_processed = preprocess(image_hat_s)

    #       in_layer = self.resnet_50_in_layers[0]
    #       out_layers = [l + ':0' for l in self.resnet_50_out_layers]
    #       image_activations = tf.graph_util.import_graph_def(
    #                                 self.resnet50_graph,
    #                                 return_elements=out_layers,
    #                                 input_map={in_layer: image_processed})
    #       image_hat_activations = tf.graph_util.import_graph_def(
    #                                 self.resnet50_graph,
    #                                 return_elements=out_layers,
    #                                 input_map={in_layer: image_hat_processed})
          
    #       act_pairs = list(zip(image_activations, image_hat_activations))
    #         # Ordered from lower to higher in the resnet50 network
    #       for i, (im_t, im_h_t) in enumerate(act_pairs):
    #         def to_name(t):
    #           return t.name.replace('/', '_').replace(':', '_')
            
    #         t_loss = tf.losses.absolute_difference(
    #                     tf.contrib.layers.flatten(im_t), 
    #                     tf.contrib.layers.flatten(im_h_t),
    #                     weights=tf.expand_dims(cweights, axis=-1))

    #         # Give lower layers higher loss to put preference on more basic
    #         # image statistics
    #         t_loss *= scale * (1.8 ** (len(act_pairs) - i))

    #         tf.summary.scalar('loss/' + to_name(im_t), t_loss)
    #         recon_image_loss += t_loss

    #         tf.summary.histogram('image/' + to_name(im_t), im_t)
    #         tf.summary.histogram('image_hat/' + to_name(im_h_t), im_h_t)

    # tf.summary.scalar('recon_image_loss', recon_image_loss)
    
    ## Total Loss
    total_loss = (
      0.000001 * total_latent_loss +
      1.000000 * multiclass_loss 
      # 0.01000 * recon_image_loss
      )
    tf.summary.scalar('total_loss', total_loss)
    
    ### Extra summaries
    # for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
    #   class_rows = tf.equal(labels, class_id)
    #   class_labels = tf.boolean_mask(labels, class_rows)
    #   class_preds = tf.boolean_mask(preds, class_rows)

    #   class_recall = tf.metrics.accuracy(
    #                       labels=tf.equal(class_labels, class_id),
    #                       predictions=tf.equal(class_preds, class_id))
    #   tf.summary.scalar('class_recall/' + class_name, class_recall)

      # tf.summary.scalar('train_labels_support/' + class_name, tf.reduce_sum(class_labels))
      # tf.summary.scalar('train_preds_support/' + class_name, tf.reduce_sum(class_preds))
      # tf.summary.histogram('train_labels_support_dist/' + class_name, class_labels)
      # tf.summary.histogram('train_preds_support_dist/' + class_name, class_preds)

      # Monitor Supervised
      # class_true = tf.boolean_mask(features, class_labels)
      # class_pred = tf.boolean_mask(features, class_preds)
      # tf.summary.image('supervised_labels', class_true, max_outputs=3, family=class_name)
      # tf.summary.image('supervised_pred', class_pred, max_outputs=3, family=class_name)

    #   # Monitor VAE
    #   Y_SHAPE = list(FEATURES_SHAPE)
    #   Y_SHAPE[0] = -1
    #   y_unflat = tf.reshape(y, Y_SHAPE)
    #   y_true = tf.boolean_mask(y_unflat, class_labels)
    #   y_pred = tf.boolean_mask(y_unflat, class_preds)
    #   tf.summary.image('vae_labels', y_true, max_outputs=3, family=class_name)
    #   tf.summary.image('vae_pred', y_pred, max_outputs=3, family=class_name)


    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          'classes': preds, #tf.argmax(logits, axis=1),
          'probabilities': tf.nn.softmax(logits),
      }
      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs={
              'classify': tf.estimator.export.PredictOutput(predictions)
          })
    elif mode == tf.estimator.ModeKeys.TRAIN:
      LEARNING_RATE = 1e-3
      
      loss = total_loss
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE, epsilon=1)
      global_step = tf.train.get_or_create_global_step()
      train_op = optimizer.minimize(loss, global_step)
      # train_op = tf.contrib.layers.optimize_loss(
      #   loss, global_step, learning_rate=LEARNING_RATE, optimizer='SGD',
      #   summaries=["gradients"])
      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.TRAIN, loss=loss, train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:

      eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(labels=labels, predictions=preds),
      }
      for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
        class_labels = tf.cast(tf.equal(labels, class_id), tf.int64)
        class_preds = tf.cast(tf.equal(preds, class_id), tf.int64)

        tf.summary.scalar('labels_support/' + class_name, tf.reduce_sum(class_labels))
        tf.summary.scalar('preds_support/' + class_name, tf.reduce_sum(class_preds))
        tf.summary.histogram('labels_support_dist/' + class_name, class_labels)
        tf.summary.histogram('preds_support_dist/' + class_name, class_preds)

        metric_kwargs = {
          'labels': class_labels,
          'predictions': class_preds,
        }
        name_to_ftor = {
          'accuracy': tf.metrics.accuracy,
          'auc': tf.metrics.auc,
          'precision': tf.metrics.precision,
          'recall': tf.metrics.recall,
        }
        for name, ftor in name_to_ftor.items():
          full_name = name + '/' + class_name
          eval_metric_ops[full_name] = ftor(name=full_name, **metric_kwargs)

      # Sadly we need to force this tensor to be computed in eval
      logging_hook = tf.train.LoggingTensorHook(
        {"obs_str_tensor" : obs_str_tensor}, every_n_iter=1)
      
      # Sadly we need to do this explicitly for eval
      summary_hook = tf.train.SummarySaverHook(
            save_secs=3,
            output_dir='/tmp/av_mobilenet_test/eval',
            scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
      hooks = [logging_hook, summary_hook]

      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.EVAL,
          loss=loss,
          eval_metric_ops=eval_metric_ops,
          evaluation_hooks=hooks)









class model_fn_simple_ff_tpu(object):

  def __init__(self, au_params):
    self.params = au_params
  
  def __call__(self, features, labels, mode, params):

    # features.set_shape([None, 170, 170, 3]) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # labels.set_shape([None,])

    tensors_to_summarize = {}
    def reg_to_summarize(v):
      def var_name(vv):
        """Magic: get the name of the variable that the caller passed to 
        `tf_variable_summaries()`"""
        import inspect
        lcls = inspect.stack()[2][0].f_locals
        for name in lcls:
          if id(vv) == id(lcls[name]):
            return name
        return None
      tensors_to_summarize[var_name(v)] = v

    # obs_str_tensor = util.ThruputObserver.monitoring_tensor('features', features)
    # just not supported on TPU ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    features = tf.cast(features, tf.float32) / 128. - 1

    # tf.contrib.summary.histogram('labels', labels)
    reg_to_summarize(labels)
    # tf.contrib.summary.histogram('features', features)
    # reg_to_summarize(features)

    from nets.mobilenet import mobilenet_v2
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
      # See also e.g. mobilenet_v2_035
      logits, endpoints = mobilenet_v2.mobilenet(
                            features,
                            num_classes=len(AV_OBJ_CLASS_NAME_TO_ID),
                            is_training=is_training,
                            depth_multiplier=self.params.DEPTH_MULTIPLIER,
                            finegrain_classification_mode=self.params.FINE)
      
      # TODO: use keras or be picky about variables to restore
      # def scaffold_fn():
      #   """Loads pretrained model through scaffold function."""
        
      #   # Per authors: Restore using exponential moving average since it produces
      #   # (1.5-2%) higher accuracy
      #   ema = tf.train.ExponentialMovingAverage(0.999)
      #   vs = ema.variables_to_restore()

      #   import os
      #   util.download(
      #     self.params.CHECKPOINT_TARBALL_URI,
      #     self.params.MODEL_BASEDIR)
      #   checkpoint = os.path.join(
      #     self.params.MODEL_BASEDIR,
      #     self.params.CHECKPOINT + '.ckpt')
        
      #   tf.train.init_from_checkpoint(
      #     checkpoint, {'MobilenetV2/' : 'MobilenetV2/'})
      #   return tf.train.Scaffold()
      scaffold_fn = None
    
    # Downweight background class
    class_weights = np.ones(len(AV_OBJ_CLASS_NAME_TO_ID))
    class_weights[0] *= .1
    class_weights /= class_weights.sum()
    class_weights = tf.cast(class_weights, tf.float32)

    weights = tf.gather(class_weights, labels)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)
    preds_scores = endpoints['Predictions']
    preds = tf.argmax(preds_scores, axis=1)
    reg_to_summarize(preds)

    for k, v in endpoints.items():
      KS = ('layer_19', 'Predictions', 'Logits')
      if any(kval in k for kval in KS):
        # tf.contrib.summary.histogram('mobilenet_' + k, v)
        tensors_to_summarize['mobilenet_' + k] = v

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          'classes': preds, #tf.argmax(logits, axis=1),
          'probabilities': tf.nn.softmax(logits),
      }
      return tf.contrib.tpu.TPUEstimatorSpec(
      # return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs={
              'classify': tf.estimator.export.PredictOutput(predictions)
          })
    elif mode == tf.estimator.ModeKeys.TRAIN:
      LEARNING_RATE = 1e-4
      
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
      optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)
      # If we are running multi-GPU, we need to wrap the optimizer.
      #if params_dict.get('multi_gpu'):
      # optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

      
      accuracy_metric = tf.metrics.accuracy(labels=labels, predictions=preds)

      # Name tensors to be logged with LoggingTensorHook.
      tf.identity(LEARNING_RATE, 'learning_rate')
      tf.identity(loss, 'cross_entropy')
      accuracy = tf.identity(accuracy_metric[1], name='train_accuracy')
      reg_to_summarize(accuracy)

      
      global_step = tf.train.get_or_create_global_step()
      # tf.contrib.summary.scalar('global_step', global_step)

      tf.contrib.summary.histogram('train_loss', loss)
      reg_to_summarize(loss)
      # tf.contrib.summary.histogram('logits', logits)

      # for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
      #   class_labels = tf.cast(tf.equal(labels, class_id), tf.int32) # ~~~~~~~~~~~
      #   class_preds = tf.cast(tf.equal(preds, class_id), tf.int32)

      #   tf.contrib.summary.scalar('train_labels_support/' + class_name, tf.reduce_sum(class_labels))
      #   tf.contrib.summary.scalar('train_preds_support/' + class_name, tf.reduce_sum(class_preds))
      #   tf.contrib.summary.histogram('train_labels_support_dist/' + class_name, class_labels)
      #   tf.contrib.summary.histogram('train_preds_support_dist/' + class_name, class_preds)

      #   class_true = tf.boolean_mask(features, class_labels)
      #   class_pred = tf.boolean_mask(features, class_preds)
      #   tf.contrib.summary.image('train_true/' + class_name, class_true, max_images=10)
      #   tf.contrib.summary.image('train_pred/' + class_name, class_pred, max_images=10)

      summ_name_tensor = sorted(tensors_to_summarize.items())
      summ_names = [n for n, t in summ_name_tensor]
      def host_call_fn(*args):
        # Host call fns are executed FLAGS.iterations_per_loop times after one
        # TPU loop is finished, setting max_queue value to the same as number of
        # iterations will make the summary writer only flush the data to storage
        # once per loop.
        with (tf.contrib.summary.create_file_writer(
            self.params.GS_MODEL_DIR,
            max_queue=10).as_default()):
          with tf.contrib.summary.always_record_summaries():
            for prefix, v in zip(summ_names, args):
              # Hack of `util.tf_variable_summaries()` for TPUs
              with tf.variable_scope(prefix):
                mean = tf.reduce_mean(v)
                tf.contrib.summary.scalar('mean', mean)
                with tf.name_scope('stddev'):
                  stddev = tf.sqrt(tf.reduce_mean(tf.square(v - mean)))
                tf.contrib.summary.scalar('stddev', stddev)
                tf.contrib.summary.scalar('max', tf.reduce_max(v))
                tf.contrib.summary.scalar('min', tf.reduce_min(v))
                tf.contrib.summary.histogram('histogram', v)
            return tf.contrib.summary.all_summary_ops()

      # To log the loss, current learning rate, and epoch for Tensorboard, the
      # summary op needs to be run on the host CPU via host_call. host_call
      # expects [batch_size, ...] Tensors, thus reshape to introduce a batch
      # dimension. These Tensors are implicitly concatenated to
      # [params['batch_size']].
      tensors_to_summarize = [
        tf.cast(tf.reshape(t, [1, -1]), tf.float32) for _, t in summ_name_tensor
      ]

      return tf.contrib.tpu.TPUEstimatorSpec(
      # return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.TRAIN,
          loss=loss,
          host_call=(host_call_fn, tensors_to_summarize),
          train_op=optimizer.minimize(loss, global_step),
          scaffold_fn=scaffold_fn)

    elif mode == tf.estimator.ModeKeys.EVAL:
      
      # preds = tf.argmax(logits, axis=1)
      classes = labels

      tf.contrib.summary.histogram('eval_preds', preds)
      tf.contrib.summary.histogram('logits', logits)

      accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)
      tf.contrib.summary.scalar('eval_accuracy', accuracy[1])
      tf.contrib.summary.scalar('eval_loss', loss)

      eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(labels=labels, predictions=preds),
      }
      for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
        class_labels = tf.cast(tf.equal(classes, class_id), tf.int64)
        class_preds = tf.cast(tf.equal(preds, class_id), tf.int64)

        tf.contrib.summary.scalar('labels_support/' + class_name, tf.reduce_sum(class_labels))
        tf.contrib.summary.scalar('preds_support/' + class_name, tf.reduce_sum(class_preds))
        tf.contrib.summary.histogram('labels_support_dist/' + class_name, class_labels)
        tf.contrib.summary.histogram('preds_support_dist/' + class_name, class_preds)

        metric_kwargs = {
          'labels': class_labels,
          'predictions': class_preds,
        }
        name_to_ftor = {
          'accuracy': tf.metrics.accuracy,
          'auc': tf.metrics.auc,
          'precision': tf.metrics.precision,
          'recall': tf.metrics.recall,
        }
        for name, ftor in name_to_ftor.items():
          full_name = name + '/' + class_name
          eval_metric_ops[full_name] = ftor(name=full_name, **metric_kwargs)


      logging_hook = tf.train.LoggingTensorHook(
        {"obs_str_tensor" : obs_str_tensor}, every_n_iter=1)
      # summary_hook = tf.train.SummarySaverHook(
      #       save_secs=3,
      #       output_dir='/tmp/av_mobilenet_test/eval',
      #       scaffold=tf.train.Scaffold(summary_op=tf.contrib.summary.merge_all()))
      hooks = [logging_hook]#, summary_hook]

      return tf.contrib.tpu.TPUEstimatorSpec(
      # return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.EVAL,
          loss=loss,
          eval_metric_ops=eval_metric_ops,
          evaluation_hooks=hooks)































class model_fn_simple_ff(object):

  def __init__(self, au_params):
    self.params = au_params
  
  def __call__(self, features, labels, mode, params):

    tf.contrib.summary.always_record_summaries()

    features.set_shape([None, 170, 170, 3]) # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    labels.set_shape([None,])

    obs_str_tensor = util.ThruputObserver.monitoring_tensor('features', features)

    features = tf.cast(features, tf.float32) / 128. - 1

    tf.summary.histogram('labels', labels)
    tf.summary.histogram('features', features)

    from nets.mobilenet import mobilenet_v2
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    with tf.contrib.slim.arg_scope(mobilenet_v2.training_scope(is_training=is_training)):
      # See also e.g. mobilenet_v2_035
      logits, endpoints = mobilenet_v2.mobilenet(
                            features,
                            num_classes=len(AV_OBJ_CLASS_NAME_TO_ID),
                            is_training=is_training,
                            depth_multiplier=self.params.DEPTH_MULTIPLIER,
                            finegrain_classification_mode=self.params.FINE)
    
    # Downweight background class
    class_weights = np.ones(len(AV_OBJ_CLASS_NAME_TO_ID))
    class_weights[0] *= .1
    class_weights /= class_weights.sum()

    weights = tf.gather(class_weights, labels)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits, weights=weights)
    preds_scores = endpoints['Predictions']
    preds = tf.argmax(preds_scores, axis=1)

    for k, v in endpoints.items():
      KS = ('layer_19', 'Predictions', 'Logits')
      if any(kval in k for kval in KS):
        tf.summary.histogram('mobilenet_' + k, v)

    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
          'classes': preds, #tf.argmax(logits, axis=1),
          'probabilities': tf.nn.softmax(logits),
      }
      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          export_outputs={
              'classify': tf.estimator.export.PredictOutput(predictions)
          })
    elif mode == tf.estimator.ModeKeys.TRAIN:
      LEARNING_RATE = 1e-4
      
      optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)

      # If we are running multi-GPU, we need to wrap the optimizer.
      #if params_dict.get('multi_gpu'):
      # optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

      
      accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)

      # Name tensors to be logged with LoggingTensorHook.
      tf.identity(LEARNING_RATE, 'learning_rate')
      tf.identity(loss, 'cross_entropy')
      tf.identity(accuracy[1], name='train_accuracy')

      # Save accuracy scalar to Tensorboard output.
      tf.summary.scalar('train_accuracy', accuracy[1])
      
      global_step = tf.train.get_or_create_global_step()
      tf.summary.scalar('global_step', global_step)

      tf.summary.histogram('train_loss', loss)
      tf.summary.histogram('logits', logits)

      for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
        class_labels = tf.cast(tf.equal(labels, class_id), tf.int64)
        class_preds = tf.cast(tf.equal(preds, class_id), tf.int64)

        tf.summary.scalar('train_labels_support/' + class_name, tf.reduce_sum(class_labels))
        tf.summary.scalar('train_preds_support/' + class_name, tf.reduce_sum(class_preds))
        tf.summary.histogram('train_labels_support_dist/' + class_name, class_labels)
        tf.summary.histogram('train_preds_support_dist/' + class_name, class_preds)

        class_true = tf.boolean_mask(features, class_labels)
        class_pred = tf.boolean_mask(features, class_preds)
        tf.summary.image('train_true/' + class_name, class_true, max_outputs=10)
        tf.summary.image('train_pred/' + class_name, class_pred, max_outputs=10)

      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.TRAIN,
          loss=loss,
          train_op=optimizer.minimize(loss, global_step))

    elif mode == tf.estimator.ModeKeys.EVAL:
      
      # preds = tf.argmax(logits, axis=1)
      classes = labels

      tf.summary.histogram('eval_preds', preds)
      tf.summary.histogram('logits', logits)

      accuracy = tf.metrics.accuracy(labels=labels, predictions=preds)
      tf.summary.scalar('eval_accuracy', accuracy[1])
      tf.summary.scalar('eval_loss', loss)

      eval_metric_ops = {
        'accuracy':
            tf.metrics.accuracy(labels=labels, predictions=preds),
      }
      for class_name, class_id in AV_OBJ_CLASS_NAME_TO_ID.items():
        class_labels = tf.cast(tf.equal(classes, class_id), tf.int64)
        class_preds = tf.cast(tf.equal(preds, class_id), tf.int64)

        tf.summary.scalar('labels_support/' + class_name, tf.reduce_sum(class_labels))
        tf.summary.scalar('preds_support/' + class_name, tf.reduce_sum(class_preds))
        tf.summary.histogram('labels_support_dist/' + class_name, class_labels)
        tf.summary.histogram('preds_support_dist/' + class_name, class_preds)

        metric_kwargs = {
          'labels': class_labels,
          'predictions': class_preds,
        }
        name_to_ftor = {
          'accuracy': tf.metrics.accuracy,
          'auc': tf.metrics.auc,
          'precision': tf.metrics.precision,
          'recall': tf.metrics.recall,
        }
        for name, ftor in name_to_ftor.items():
          full_name = name + '/' + class_name
          eval_metric_ops[full_name] = ftor(name=full_name, **metric_kwargs)


      logging_hook = tf.train.LoggingTensorHook(
        {"obs_str_tensor" : obs_str_tensor}, every_n_iter=1)
      summary_hook = tf.train.SummarySaverHook(
            save_secs=3,
            output_dir='/tmp/av_mobilenet_test/eval',
            scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
      hooks = [logging_hook, summary_hook]

      return tf.estimator.EstimatorSpec(
          mode=tf.estimator.ModeKeys.EVAL,
          loss=loss,
          eval_metric_ops=eval_metric_ops,
          evaluation_hooks=hooks)






def main_tpu():

  import os
  assert 'TPU_NAME' in os.environ

  import time
  model_dir = 'gs://au2018-3/avmobilenet_tpu_test/test_%s' % time.time()

  tf.logging.set_verbosity(tf.logging.INFO)

  tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
    coordinator_name='coordinator')
    # tpu=str(os.environ['TPU_NAME']))
  
  # from tensorflow.python.training import server_lib
  # coord = server_lib.Server.create_local_server()
  # coord = tf.train.ClusterSpec({
  #   "coordinator": ["localhost:2222", "localhost:2223"]})

  # import pdb; pdb.set_trace()

  TPU_ITERATIONS = 50
  BATCH_SIZE = 4096
  # TPU_CORES = 8
  # train_distribution = tf.contrib.distribute.TPUStrategy(
  #     tpu_cluster_resolver, steps_per_run=BATCH_SIZE * TPU_CORES)
  config = tf.contrib.tpu.RunConfig(
      model_dir=model_dir,
      save_summary_steps=10,
      save_checkpoints_secs=10,
      log_step_count_steps=10,
      cluster=tpu_cluster_resolver,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      # train_distribute=train_distribution,
      # eval_distribute=train_distribution)
      tpu_config=tf.contrib.tpu.TPUConfig(TPU_ITERATIONS))


  from au.fixtures.tf.mobilenet import Mobilenet
  params = Mobilenet.Medium()
  params.GS_MODEL_DIR = model_dir
  # params.BATCH_SIZE = BATCH_SIZE
  av_classifier = tf.contrib.tpu.TPUEstimator(
    model_fn=model_fn_simple_ff_tpu(params),
    use_tpu=True,
    params=None,
    config=config,
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    predict_batch_size=BATCH_SIZE)
  # av_classifier = tf.estimator.Estimator(
  #   model_fn=model_fn_simple_ff_tpu(params),
  #   # use_tpu=True,
  #   params=None,
  #   config=config)
  #   # train_batch_size=BATCH_SIZE,
  #   # eval_batch_size=BATCH_SIZE,
  #   # predict_batch_size=BATCH_SIZE)

  with Spark.getOrCreate() as spark:
    
    # df = spark.read.parquet('cache/data/argoverse_cropped_object_170_170/')
    df = spark.read.parquet('gs://au2018-3/crops_full/argoverse_cropped_object_170_170')
    # partition discovery is quite slow!!!!!!! :( ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # print('num images', df.count())

    def to_example(row):
      import imageio
      from io import BytesIO
      img = imageio.imread(BytesIO(row.jpeg_bytes))
      label = AV_OBJ_CLASS_NAME_TO_ID[row.category_name]
      # # For tpu
      # img = img.astype(float)
      return img, label
    
    # TPU Estimator needs static shapes :P
    def set_shapes(ds):
      def _set_shapes(images, labels):
        images.set_shape(images.get_shape().merge_with(
            tf.TensorShape([BATCH_SIZE, 170, 170, 3])))
        labels.set_shape(labels.get_shape().merge_with(
            tf.TensorShape([BATCH_SIZE])))
        return images, labels
      
      ds = ds.map(_set_shapes)
      return ds
      

    tdf = df.filter(df.split == 'train')
    def train_input_fn(params=None):
      with tf.device('/job:coordinator/task:0'):
        train_ds = spark_df_to_tf_dataset(tdf, 'shard', to_example, (tf.uint8, tf.int32), logging_name='train')
        
        def to_bfloat(images, labels):
          images = tf.cast(images, tf.bfloat16)
          return images, labels
        train_ds = train_ds.map(to_bfloat)
        # train_ds = train_ds.cache()
        
        train_ds = train_ds.shuffle(8000)
        train_ds = train_ds.batch(BATCH_SIZE)
        train_ds = set_shapes(train_ds)
        train_ds = train_ds.prefetch(tf.contrib.data.AUTOTUNE)
        
        source_dataset = train_ds
        source_iterator = train_ds.make_one_shot_iterator()
        source_handle = source_iterator.string_handle()


      # https://github.com/tensorflow/tensorflow/blob/76f23b3b751150a2a81012925100bb08e406ab47/tensorflow/python/tpu/datasets.py#L51
      from tensorflow.python.data.experimental.ops import batching
      from tensorflow.python.data.experimental.ops import interleave_ops
      from tensorflow.python.data.ops import dataset_ops
      from tensorflow.python.data.ops import iterator_ops
      from tensorflow.python.data.ops import readers
      from tensorflow.python.framework import dtypes
      from tensorflow.python.framework import function
      from tensorflow.python.framework import ops
      from tensorflow.python.ops import functional_ops
      @function.Defun(dtypes.string)
      def LoadingFunc(h):
        remote_iterator = iterator_ops.Iterator.from_string_handle(
            h, dataset_ops.get_legacy_output_types(source_dataset),
            dataset_ops.get_legacy_output_shapes(source_dataset))
        return remote_iterator.get_next()

      def MapFn(unused_input):
        source_dataset_output_types = dataset_ops.get_legacy_output_types(
            source_dataset)
        if isinstance(source_dataset_output_types, dtypes.DType):
          output_types = [source_dataset_output_types]
        elif isinstance(source_dataset_output_types, (list, tuple)):
          output_types = source_dataset_output_types
        else:
          raise ValueError('source dataset has invalid output types')
        remote_calls = functional_ops.remote_call(
            args=[source_handle],
            Tout=output_types,
            f=LoadingFunc,
            target='/job:%s/replica:0/task:0/cpu:0' % 'coordinator')
        if len(remote_calls) == 1:
          return remote_calls[0]
        else:
          return remote_calls

      with ops.device('/job:%s' % 'worker'):
        output_dataset = dataset_ops.Dataset.range(2).repeat().map(
            MapFn, num_parallel_calls=4)
        output_dataset = output_dataset.prefetch(1)

        
        # Undo the batching used during the transfer.
        # output_dataset = output_dataset.apply(batching.unbatch()).prefetch(1)
        
        output_dataset = set_shapes(output_dataset)

      return output_dataset


      """
      Thruput from code above looks to use about 50% of network capacity ?
      https://blog.serverdensity.com/network-performance-aws-google-rackspace-softlayer/
      estimated 3gigbit in us-central-1a

      Could be bottlnecked on TPU RPC endpoint; Tensorflow reports
      350-415 examples per second no matter how we tune Spark parallelism.

      To get max I/O, likely need to use ops that can run on worker machine.

      with float32 images batch size 4096
      2019-08-26 07:52:19,800 au   29574 : Progress for 
      train
      -------------------  ----------------------------------------------
      Thruput
      N thru               2762966
      N chunks             56
      total time           2 hours and 3.02 seconds
      total thru           1.92 TB
      rate                 266.06 MB / sec
      Hz                   383.58443542574645
      Progress
      Percent complete     56.02906977215781
      ETA                  1 hour, 34 minutes and 12.84 seconds
      Latency (per chunk)
      avg                  2 minutes, 8 seconds and 625.34 milliseconds
      p50                  18 seconds and 211.27 milliseconds
      p95                  14 minutes, 32 seconds and 33.23 milliseconds
      p99                  17 minutes, 13 seconds and 954.11 milliseconds
      -------------------  ----------------------------------------------


      with bfloat16 images; looks tpu bound! CPU sits idle sometimes on 
      coodinator.  batch size 4096
      INFO:tensorflow:global_step/sec: 0.124113
      INFO:tensorflow:examples/sec: 508.366
      2019-08-26 09:23:34,392 au   6879 : Progress for
      train
      -------------------  ----------------------------------------------
      Thruput
      N thru               2368435
      N chunks             48
      total time           1 hour, 14 minutes and 11.86 seconds
      total thru           205.4 GB
      rate                 46.14 MB / sec
      Hz                   532.0107497600044
      Progress
      Percent complete     48.0285352283816
      ETA                  1 hour, 20 minutes and 17.33 seconds
      Latency (per chunk)
      avg                  1 minute, 32 seconds and 746.99 milliseconds
      p50                  7 seconds and 331.97 milliseconds
      p95                  11 minutes, 10 seconds and 406.4 milliseconds
      p99                  12 minutes, 10 seconds and 253.85 milliseconds
      -------------------  ----------------------------------------------


      launch is slower with gs:// (instead of local) read, but thruput is same:
      INFO:tensorflow:loss = 0.027581474, step = 200 (411.267 sec)
      INFO:tensorflow:global_step/sec: 0.121576
      INFO:tensorflow:examples/sec: 497.974
      2019-08-26 10:49:12,658 au   16962 : Progress for 
      train
      -------------------  ---------------------------------------------
      Thruput
      N thru               793686
      N chunks             16
      total time           21 minutes and 53.8 seconds
      total thru           68.83 GB
      rate                 52.39 MB / sec
      Hz                   604.1145149510429
      Progress
      Percent complete     16.094837312940097
      ETA                  1 hour, 54 minutes and 9.07 seconds
      Latency (per chunk)
      avg                  1 minute, 22 seconds and 112.54 milliseconds
      p50                  5 seconds and 573.17 milliseconds
      p95                  9 minutes, 28 seconds and 113.97 milliseconds
      p99                  11 minutes, 17 seconds and 522.4 milliseconds
      -------------------  ---------------------------------------------
      """


        # tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS,
        #                      iterator.initializer)
        # batch = iterator.get_next()
        # return batch

        # return train_ds

    edf = df.filter(df.split == 'val')
    def eval_input_fn(params=None):
      eval_ds = spark_df_to_tf_dataset(edf, 'shard', to_example, (tf.float32, tf.int32), logging_name='test')
      
      eval_ds = eval_ds.batch(BATCH_SIZE)
      eval_ds = set_shapes(eval_ds)

      eval_ds = eval_ds.take(100)

      return eval_ds


    TRAIN_EPOCHS = 100

    is_training = [True]
    def run_eval():
      import time
      time.sleep(120)
      while is_training[0]:
        util.log.info("Running eval ...")
        # eval_config = config.replace(session_config=util.tf_cpu_session_config())
        # eval_av_classifier = tf.estimator.Estimator(
        #                         model_fn=model_fn(params),
        #                         params=None,
        #                         config=eval_config)
        # eval_results = eval_av_classifier.evaluate(input_fn=eval_input_fn)#, hooks=[summary_hook])
        # util.log.info('\nEvaluation results:\n\t%s\n' % eval_results)
    
    import threading
    eval_thread = threading.Thread(target=run_eval)
    # eval_thread.start()

    for t in range(TRAIN_EPOCHS):
      av_classifier.train(input_fn=train_input_fn, max_steps=int(1e7))
      
      #, hooks=train_hooks)
      # if t % 10 == 0 or t >= TRAIN_EPOCHS - 1:
      # summary_hook = tf.train.SummarySaverHook(
      #       save_secs=3,
      #       output_dir='/tmp/av_mobilenet_test/eval',
      #       scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
      # eval_results = av_classifier.evaluate(input_fn=eval_input_fn)#, hooks=[summary_hook])
      # util.log.info('\nEvaluation results:\n\t%s\n' % eval_results)
    is_training[0] = False
    # eval_thread.join()









def main():

  tf_config = util.tf_create_session_config()
  # tf_config = util.tf_cpu_session_config()

  model_dir = '/tmp/av_mobilenet_test'
  util.mkdir(model_dir)

  #import os
  #import json
  #os.environ['TF_CONFIG'] = json.dumps({
  #  "cluster": {
  #    "worker": ["localhost:1122", "localhost:1123"],
#	  "chief": ["localhost:1120"],
#    },
#    'task': {'type': 'worker', 'index': 0}
#  })

  config = tf.estimator.RunConfig(
             model_dir=model_dir,
             save_summary_steps=10,
             save_checkpoints_secs=10,
             session_config=tf_config,
             log_step_count_steps=10)
  gpu_dist = tf.contrib.distribute.MirroredStrategy()
  # cpu_dist = tf.contrib.distribute.MirroredStrategy(devices=['/cpu:0'])
  
  config = config.replace(train_distribute=gpu_dist)#, eval_distribute=gpu_dist)#cpu_dist)

  from au.fixtures.tf.mobilenet import Mobilenet
  params = Mobilenet.Medium()
  av_classifier = tf.estimator.Estimator(
    # model_fn=model_fn_simple_ff(params),
    model_fn=model_fn_vae_hybrid(params),
    params=None,
    config=config)

  with Spark.getOrCreate() as spark:
    
    # if util.missing_or_empty('/tmp/balanced_sample'):
    #   df = spark.read.parquet('/opt/au/cache/argoverse_cropped_object_170_170_small')
    #   # df = spark.read.parquet('/outer_root/media/seagates-ext4/au_datas/crops_full/argoverse_cropped_object_170_170/')#'/outer_root/media/seagates-ext4/au_datas/crops_full/argoverse_cropped_object_170_170/')#'/opt/au/cache/argoverse_cropped_object_170_170')
    #   from au.spark import get_balanced_sample
    #   categories = [
    #     "background",
    #     "VEHICLE",
    #     "PEDESTRIAN",
    #   ]
    #   df = df.filter(df.category_name.isin(categories))
    #   fair_df = get_balanced_sample(df, 'category_name', n_per_category=5000)
      
    #   # Re-shard
    #   import pyspark.sql.functions as F
    #   fair_df = fair_df.withColumn(
    #             'fair_shard',
    #             F.abs(F.hash(fair_df['uri'])) % 2)
    #   fair_df = fair_df.select(*list(set(fair_df.columns) - set(['shard'])))
    #   fair_df = fair_df.withColumn('shard', fair_df['fair_shard'])

    #   fair_df.write.parquet(
    #     '/tmp/balanced_sample',
    #     compression='none',
    #     partitionBy=['split', 'shard'])
    #   print("wrote to ", '/tmp/balanced_sample')
    
    # df = spark.read.parquet('/tmp/balanced_sample')
    
    df = spark.read.parquet('/outer_root/media/seagates-ext4/au_datas/crops_full/argoverse_cropped_object_170_170/')#'/outer_root/media/seagates-ext4/au_datas/crops_full/argoverse_cropped_object_170_170/')#'/opt/au/cache/argoverse_cropped_object_170_170')
    print('num images', df.count())

    def to_example(row):
      import imageio
      from io import BytesIO
      img = imageio.imread(BytesIO(row.jpeg_bytes))
      label = AV_OBJ_CLASS_NAME_TO_ID[row.category_name]
      return img, label
   
    # BATCH_SIZE = 300
    BATCH_SIZE = 20
    tdf = df.filter(df.split == 'train')
    def train_input_fn():
      train_ds = spark_df_to_tf_dataset(tdf, 'shard', to_example, (tf.uint8, tf.int64), logging_name='train')
      
      # train_ds = train_ds.cache()
      train_ds = train_ds.repeat(10)
      train_ds = train_ds.shuffle(100)
      train_ds = train_ds.batch(BATCH_SIZE)
      

      # train_ds = train_ds.take(5)

      
      # train_ds = add_stats(train_ds)
      return train_ds

    edf = df.filter(df.split == 'val')#spark.createDataFrame(df.filter(df.split == 'val').take(3000))
    def eval_input_fn():
      eval_ds = spark_df_to_tf_dataset(edf, 'shard', to_example, (tf.uint8, tf.int64), logging_name='test')
      eval_ds = eval_ds.batch(BATCH_SIZE)

      eval_ds = eval_ds.take(100)

      # eval_ds = add_stats(eval_ds)
      return eval_ds

    # # Set up hook that outputs training logs every 100 steps.
    # from official.utils.logs import hooks_helper
    # train_hooks = hooks_helper.get_train_hooks(
    #    ['ExamplesPerSecondHook',
    #    'LoggingTensorHook'],
    #    model_dir=model_dir,
    #    batch_size=BATCH_SIZE)#params.BATCH_SIZE)

    TRAIN_EPOCHS = 100
    #train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=TRAIN_EPOCHS)
    #eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100, throttle_secs=120)
    #tf.estimator.train_and_evaluate(av_classifier, train_spec, eval_spec)


    ## Train and evaluate model.
    #TRAIN_EPOCHS = 1000

    is_training = [True]
    def run_eval():
      import time
      time.sleep(120)
      while is_training[0]:
        util.log.info("Running eval ...")
        eval_config = config.replace(session_config=util.tf_cpu_session_config())
        eval_av_classifier = tf.estimator.Estimator(
                                model_fn=model_fn(params),
                                params=None,
                                config=eval_config)
        eval_results = eval_av_classifier.evaluate(input_fn=eval_input_fn)#, hooks=[summary_hook])
        util.log.info('\nEvaluation results:\n\t%s\n' % eval_results)
    
    import threading
    eval_thread = threading.Thread(target=run_eval)
    # eval_thread.start()

    for t in range(TRAIN_EPOCHS):
      av_classifier.train(input_fn=train_input_fn)
      
      #, hooks=train_hooks)
      # if t % 10 == 0 or t >= TRAIN_EPOCHS - 1:
      # summary_hook = tf.train.SummarySaverHook(
      #       save_secs=3,
      #       output_dir='/tmp/av_mobilenet_test/eval',
      #       scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
      # eval_results = av_classifier.evaluate(input_fn=eval_input_fn)#, hooks=[summary_hook])
      # util.log.info('\nEvaluation results:\n\t%s\n' % eval_results)
    is_training[0] = False
    # eval_thread.join()


if __name__ == '__main__':
  main_tpu()
