from au import conf
from au.fixtures import dataset
from au.fixtures import nnmodel
from au.test import testconf
from au.test import testutils

import os
import unittest

import pytest

import tensorflow as tf

class Sobel(nnmodel.INNModel):
  
  def __init__(self, params=None):
    self.params = params or Sobel.Params()

  class Params(nnmodel.INNModel.ParamsBase):
    def __init__(self):
      super(Sobel.Params, self).__init__(model_name='Sobel')
      self.INPUT_TENSOR_SHAPE = [None, 200, 300, 3]
    
  class GraphFactory(nnmodel.TFInferenceGraphFactory):
    def create_inference_graph(self, input_image, base_graph):
      with base_graph.as_default():
        # FMI see impl https://github.com/tensorflow/tensorflow/blob/r1.11/tensorflow/python/ops/image_ops_impl.py#L2770
        sobel = tf.image.sobel_edges(tf.cast(input_image, tf.float32))
        self.out = tf.identity(sobel, name='sobel')
      return base_graph
    
    @property
    def output_names(self):
      return ('sobel:0',)
  
  @classmethod
  def load_or_train(cls, params=None):
    return Sobel(params=params)
  
  def get_inference_graph(self):
    return Sobel.GraphFactory(params=self.params)


def _create_fixture(monkeypatch):
  TEST_TEMPDIR = os.path.join(
                      testconf.TEST_TEMPDIR_ROOT,
                      'sobel_test')
  testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)

  dataset.ImageTable.setup()
  
  model = Sobel.load_or_train()
  filler = nnmodel.FillActivationsTFDataset(model=model)

  rows = list(dataset.ImageTable.iter_all_rows())
  assert rows

  class Fixture(object):
    pass
  fixture = Fixture()
  fixture.rows = rows
  fixture.filler = filler
  fixture.model = model
  return fixture

def _check_rows(fixture, filled_rows):
  assert len(filled_rows)
  assert len(fixture.rows) == len(filled_rows)
  
  for row in filled_rows:
    assert row.attrs is not ''
    assert 'activations' in row.attrs
    
    igraph = fixture.model.get_inference_graph()

    acts = row.attrs['activations']
    assert acts
    act = acts[0]
    assert act.model_name == igraph.model_name
    
    import imageio
    import numpy as np
    
    if '202228408_eccfe4790e' in row.uri:
    
      tensor_name = igraph.output_names[0]
      sobel_tensor = act.tensor_to_value[tensor_name]
      
      sobel_y = sobel_tensor[...,0]
      sobel_x = sobel_tensor[...,1]
      
      # We need to compare actual and expected via image bytes b/c imageio
      # does some sort of subtle color normalization and we want our fixtures
      # to simply be PNGs.
      def to_png_bytes(arr):
        import io
        buf = io.BytesIO()
        imageio.imwrite(buf, arr, 'png')
        return buf.getvalue()
      
      sobel_y_bytes = to_png_bytes(sobel_y)
      sobel_x_bytes = to_png_bytes(sobel_x)
      
      SOBEL_Y_TEST_IMG_PATH = os.path.join(
                  conf.AU_ROOT,
                  'au/test/202228408_eccfe4790e.jpg.png.sobel_y.png')
      SOBEL_X_TEST_IMG_PATH = os.path.join(
                  conf.AU_ROOT,
                  'au/test/202228408_eccfe4790e.jpg.png.sobel_x.png')
      
      assert sobel_y_bytes == open(SOBEL_Y_TEST_IMG_PATH).read()
      assert sobel_x_bytes == open(SOBEL_X_TEST_IMG_PATH).read()
    
      # For debugging
      visible_path = row.to_debug()
      imageio.imwrite(visible_path + '.sobel_x.png', sobel_x)
      imageio.imwrite(visible_path + '.sobel_y.png', sobel_y)
      print("Debug images saved to %s" % visible_path)

def test_activations_sobel(monkeypatch):
  fixture = _create_fixture(monkeypatch)

  filled = list(fixture.filler(fixture.rows))
  _check_rows(fixture, filled)
  
@pytest.mark.slow
def test_spark_activations_sobel(monkeypatch):
  fixture = _create_fixture(monkeypatch)

  with testutils.LocalSpark.sess() as spark:
    sc = spark.sparkContext
    rdd = sc.parallelize(fixture.rows)
    rdd = rdd.mapPartitions(fixture.filler)

    filled = rdd.collect()
    _check_rows(fixture, filled)

@pytest.mark.slow
def test_fill_activations_table(monkeypatch):
  fixture = _create_fixture(monkeypatch)

  class TestActivationsTable(nnmodel.ActivationsTable):
    TABLE_NAME = 'sobel_fill_activations_test'
    NNMODEL_CLS = Sobel
    IMAGE_TABLE_CLS = dataset.ImageTable

  with testutils.LocalSpark.sess() as spark:
    TestActivationsTable.setup(spark=spark)

    df = spark.read.parquet(TestActivationsTable.table_root())
    df.createOrReplaceTempView("sobel_activations")
    spark.sql("SELECT * FROM sobel_activations").show()

