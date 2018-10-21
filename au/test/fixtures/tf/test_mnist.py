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
  monkeypatch.setattr(conf, 'AU_TABLE_CACHE', os.path.join(TEST_TEMPDIR, 'tables'))
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
def test_mnist_dataset(monkeypatch):
  _setup(monkeypatch)
  
  
  
  params = mnist.MNIST.Params()
  params.LIMIT = 100
  
  mnist.MNISTDataset.init(params=params)
  
  rows = mnist.MNISTDataset.get_rows_by_uris(
                                  ('mnist_train_0',
                                   'mnist_test_0',
                                   'not_in_mnist'))
  assert len(rows) == 2
  rows = sorted(rows)
  assert rows[0].uri == 'mnist_test_0'
  assert rows[1].uri == 'mnist_train_0'
  expected_bytes = open(testconf.MNIST_TEST_IMG_PATH, 'rb').read()
  assert rows[0].image_bytes == expected_bytes


  mnist.MNISTDataset.save_datasets_as_png(params=params)
  TEST_PATH = os.path.join(
                TEST_TEMPDIR,
                'data/MNIST/test/MNIST-test-label_7-mnist_test_0.png') 
  assert os.path.exists(TEST_PATH)

  import imageio
  expected = imageio.imread(testconf.MNIST_TEST_IMG_PATH)

  import numpy as np
  image = imageio.imread(TEST_PATH)
  np.testing.assert_array_equal(image, expected)
