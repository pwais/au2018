from au import conf
from au import util
from au.fixtures import alm
from au.fixtures import dataset
from au.fixtures import nnmodel
from au.test import testconf
from au.test import testutils

import os
import unittest

import numpy as np

import pytest

TEST_TEMPDIR = os.path.join(testconf.TEST_TEMPDIR_ROOT, 'alm_test')

def _filled_mock_activations(row):
  arr = row.as_numpy()
  acts = nnmodel.Activations()
  acts.set_tensor('visible', 'img', arr)
  row.attrs = {}
  row.attrs['activations'] = acts
  return row

class TestBasicAutoencoder(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Set up our test image table
    from _pytest.monkeypatch import MonkeyPatch
    monkeypatch = MonkeyPatch()
    testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)
    dataset.ImageTable.setup()

    # Use a uniform image size
    cls.IMAGE_H = 100
    cls.IMAGE_W = 200
    cls.FILLED_ROWS = [
      _filled_mock_activations(r.resized(cls.IMAGE_H, cls.IMAGE_W))
      for r in dataset.ImageTable.iter_all_rows()
    ]

  def test_example_xform(self):
    xform = alm.ImageRowToExampleXForm()
    for row in self.FILLED_ROWS:
      ex = xform(row)
      np.testing.assert_array_equal(ex.x, ex.y.flatten())
      assert ex.uri == row.uri
    
  def test_activation_dataset(self):
    
    class MockActivationTable(nnmodel.ActivationsTable):
      @classmethod
      def as_imagerow_rdd(cls, spark=None):
        with testutils.LocalSpark.sess(spark) as spark:
          return spark.sparkContext.parallelize(self.FILLED_ROWS)
      
    class TestActivationDataset(alm.ActivationsDataset):
      ACTIVATIONS_TABLE = MockActivationTable
    
    with testutils.LocalSpark.sess() as spark:
      ds = TestActivationDataset.as_tf_dataset(spark)
      with util.tf_data_session(ds) as (sess, iter_dataset):
        ds_tuples = list(iter_dataset())

    assert len(ds_tuples) == len(self.FILLED_ROWS)
    
    for x, y, uri in ds_tuples:
      np.testing.assert_array_equal(x, y.flatten())

      # Spot-check a specific image
      test_img_fname = '2929331372_398d58807e.jpg'
      if test_img_fname in uri:
        test_img_path = os.path.join(
                          conf.AU_IMAGENET_SAMPLE_IMGS_DIR,
                          test_img_fname)
        import imageio
        expected_visible = imageio.imread(test_img_path)
        import cv2
        expected_visible = cv2.resize(
                                expected_visible,
                                (self.IMAGE_W, self.IMAGE_H))
        np.testing.assert_array_equal(y, expected_visible)
