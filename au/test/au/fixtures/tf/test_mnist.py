from au import util
from au.fixtures.tf import mnist

import pytest

MODEL_BASEDIR = '/tmp/au_test/test_mnist'

@pytest.mark.slow
def test_mnist_train():
  util.mkdir(MODEL_BASEDIR)
  util.rm_rf(MODEL_BASEDIR)
  
  params = mnist.MNistEager.Params()
  params.TRAIN_EPOCHS = 1
  params.LIMIT = 10
  params.MODEL_BASEDIR = MODEL_BASEDIR
  model = mnist.MNistEager.load_or_train(params)
  
  
  print list(model.iter_activations())
  