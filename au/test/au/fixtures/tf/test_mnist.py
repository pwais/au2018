from au import util
from au.fixtures.tf import mnist

import pytest

TEST_BASEDIR = '/tmp/au_test/test_mnist'

@pytest.mark.slow
def test_mnist_train():
  util.mkdir(TEST_BASEDIR)
  util.rm_rf(TEST_BASEDIR)
  
  params = mnist.MNistEager.Params()
  params.TRAIN_EPOCHS = 1
  params.LIMIT = 10
  params.MODEL_BASEDIR = TEST_BASEDIR
  model = mnist.MNistEager.load_or_train(params)
  
  
  print list(model.iter_activations())

@pytest.mark.slow
def test_mnist_save_pngs():
  util.mkdir(TEST_BASEDIR)
  util.rm_rf(TEST_BASEDIR)
  
  params = mnist.MNistEager.Params()
  params.DATA_BASEDIR = TEST_BASEDIR
  params.LIMIT = 10

  mnist.MNistEager.save_datasets_as_png(params)
