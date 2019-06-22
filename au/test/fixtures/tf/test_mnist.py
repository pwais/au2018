import os

from au import util
from au.fixtures.tf import mnist
from au.test import testconf
from au.test import testutils

import unittest

import imageio
import numpy as np
import pytest

TEST_TEMPDIR = os.path.join(testconf.TEST_TEMPDIR_ROOT, 'test_mnist') 

@pytest.mark.slow
def test_mnist_train(monkeypatch):
  testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)

  params = mnist.MNIST.Params()
  params.TRAIN_EPOCHS = 10
  params.LIMIT = 1000
  mnist.MNISTDataset.setup(params=params)
  model = mnist.MNIST.load_or_train(params)


class TestMNISTDataset(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Use /tmp for test fixtures
    from _pytest.monkeypatch import MonkeyPatch
    monkeypatch = MonkeyPatch()
    testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)
  
    cls.params = mnist.MNIST.Params()
    cls.params.LIMIT = 100
    
    mnist.MNISTDataset.setup(params=cls.params)
    assert os.path.exists(
                os.path.join(
                  TEST_TEMPDIR,
                  'data/MNIST/train-images-idx3-ubyte'))
    
    cls.actual_test_0_path = os.path.join(
                  TEST_TEMPDIR,
                  'data/MNIST/test/MNIST-test-label_7-mnist_test_0.png')
    cls.expected_test_0_bytes = open(testconf.MNIST_TEST_IMG_PATH, 'rb').read()
    cls.expected_test_0 = imageio.imread(testconf.MNIST_TEST_IMG_PATH)

  def _check_test_0_img(self, rows=None, tuples=None):
    tested = False
    for row in rows or []:
      if 'mnist_test_0' in row.uri:
        assert row.image_bytes == self.expected_test_0_bytes
        tested = True
        break

    for t in tuples or []:
      arr, label, uri = t
      uri = uri.decode('utf-8')
      if 'mnist_test_0' in uri:
        def normalize(im):
          from au.fixtures.dataset import ImageRow
          norm = mnist.MNIST.Params().make_normalize_ftor()
          row = norm(ImageRow.from_np_img_labels(self.expected_test_0))
          return row.attrs['normalized']
        expected = normalize(self.expected_test_0)
        assert arr.shape == expected.shape
        np.testing.assert_array_equal(arr, expected)
        tested = True
        break
    
    assert tested, "Row not found?"

  @pytest.mark.slow
  def test_get_rows(self):
    uris = (
      'mnist_train_0',
      'mnist_test_0',
      'not_in_mnist',
    )
    rows = mnist.MNISTDataset.get_rows_by_uris(uris)
    assert len(rows) == 2
    
    rows = sorted(rows)
    assert rows[0].uri == 'mnist_test_0'
    assert rows[1].uri == 'mnist_train_0'
    expected_bytes = open(testconf.MNIST_TEST_IMG_PATH, 'rb').read()
    assert rows[0].image_bytes == expected_bytes

  @pytest.mark.slow
  def test_image_contents(self):
    mnist.MNISTDataset.save_datasets_as_png(params=self.params)
    assert os.path.exists(self.actual_test_0_path)
    im = imageio.imread(self.actual_test_0_path)
    np.testing.assert_array_equal(im, self.expected_test_0)

  @pytest.mark.slow
  def test_spark_df(self):
    with testutils.LocalSpark.sess() as spark:
      df = mnist.MNISTDataset.as_imagerow_df(spark)
      df.show()
      assert df.count() == 2 * self.params.LIMIT
      self._check_test_0_img(rows=df.collect())

      df = mnist.MNISTDataset.get_class_freq(spark)
      print('MNISTDataset.get_class_freq:')
      df.show()
      for row in df.collect():
        # We should have a reasonable sample ...
        assert row.frac >= 0.01

  
  @pytest.mark.slow
  def test_to_tf_dataset_no_spark(self):
    d = mnist.MNISTDataset.to_mnist_tf_dataset()
    with util.tf_data_session(d) as (sess, iter_dataset):
      tuples = list(iter_dataset())
    assert len(tuples) == 2 * self.params.LIMIT
    self._check_test_0_img(tuples=tuples)
  
  @pytest.mark.slow
  def test_to_tf_dataset_spark(self):
    with testutils.LocalSpark.sess() as spark:
      d = mnist.MNISTDataset.to_mnist_tf_dataset(spark=spark)
      with util.tf_data_session(d) as (sess, iter_dataset):
        tuples = list(iter_dataset())
      self._check_test_0_img(tuples=tuples)



@pytest.mark.slow
def test_mnist_igraph(monkeypatch):
  from au.fixtures import nnmodel

  testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)
  
  params = mnist.MNIST.Params()
  params.TRAIN_EPOCHS = 1
  params.LIMIT = 10
  mnist.MNISTDataset.setup(params=params)
  model = mnist.MNIST.load_or_train(params)
  igraph = model.get_inference_graph()
  assert igraph != nnmodel.TFInferenceGraphFactory()

  mnist.MNISTDataset.setup(params=params)
  rows = list(mnist.MNISTDataset.iter_all_rows())

  filler = nnmodel.FillActivationsTFDataset(model=model)
  out_rows = list(filler(rows))
  assert len(out_rows) == len(rows)
  for row in out_rows:
    acts = row.attrs['activations']
    assert igraph.model_name in acts.get_models()
    for tensor_name in model.igraph.output_names:
      t = acts.get_tensor(igraph.model_name, tensor_name)
      
      # Check that we have a non-empty array
      assert t.shape
