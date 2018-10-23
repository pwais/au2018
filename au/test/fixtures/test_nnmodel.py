from au import conf
from au.fixtures import dataset
from au.fixtures import nnmodel
from au.test import testconf

import os

import tensorflow as tf

class Sobel(nnmodel.INNModel):
  
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
  
def test_activations_sobel(monkeypatch):
  
  TEST_TEMPDIR = os.path.join(
                      testconf.TEST_TEMPDIR_ROOT,
                      'sobel_test')
  testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)
  
  dataset.ImageTable.init()
  
  params = Sobel.Params()
  tigraph_factory = Sobel.GraphFactory(params)
  filler = nnmodel.FillActivationsTFDataset(tigraph_factory)
  
  irows = dataset.ImageTable.iter_all_rows()
  filled = list(filler(irows))
  
  for row in filled:
    assert row.attrs is not ''
    assert 'activation_to_val' in row.attrs
    
    activation_to_val = row.attrs['activation_to_val'] 
    
    import imageio
    import numpy as np
    
    if '202228408_eccfe4790e' in row.uri:
    
      sobel_tensor = activation_to_val[tigraph_factory.output_names[0]]
      
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
#     visible_path = row.to_debug()
#     imageio.imwrite(visible_path + '.sobel_x.png', sobel_x)
#     imageio.imwrite(visible_path + '.sobel_y.png', sobel_y)
#     print visible_path

    
    