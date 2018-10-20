import os

from au import conf
from au import util
from au.fixtures.tf import mnist
from au.test import testconf

import pytest

TEST_TEMPDIR = os.path.join(testconf.TEST_TEMPDIR_ROOT, 'test_mnist') 

def _setup(monkeypatch):
  monkeypatch.setattr(conf, 'AU_CACHE', TEST_TEMPDIR)
  monkeypatch.setattr(conf, 'AU_DATA_CACHE', os.path.join(TEST_TEMPDIR, 'data'))
  monkeypatch.setattr(conf, 'AU_MODEL_CACHE', os.path.join(TEST_TEMPDIR, 'models'))
  monkeypatch.setattr(conf, 'AU_TENSORBOARD_DIR', os.path.join(TEST_TEMPDIR, 'tensorboard'))
  util.mkdir(TEST_TEMPDIR)
  util.rm_rf(TEST_TEMPDIR)

@pytest.mark.slow
def test_mnist_train(monkeypatch):
  _setup(monkeypatch)

  params = mnist.MNIST.Params()
  params.TRAIN_EPOCHS = 1
  params.LIMIT = 10
  model = mnist.MNIST.load_or_train(params)
  
  
  print list(model.iter_activations())

@pytest.mark.slow
def test_mnist_save_pngs(monkeypatch):
  _setup(monkeypatch)
  
  params = mnist.MNIST.Params()
  params.LIMIT = 100

  mnist.MNIST.save_datasets_as_png(params)
  
  TEST_PATH = os.path.join(
                TEST_TEMPDIR,
                'data/MNIST/test/images/img_0_label-7.png') 
  
  assert os.path.exists(TEST_PATH)

  import imageio
  image = imageio.imread(TEST_PATH)
  expected = imageio.imread(testconf.MNIST_TEST_IMG_PATH)
  
  import numpy as np
  np.testing.assert_array_equal(image, expected)
