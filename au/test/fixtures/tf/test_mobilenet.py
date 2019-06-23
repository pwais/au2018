from au import util
from au.fixtures import dataset
from au.fixtures import nnmodel
from au.fixtures.tf import mobilenet
from au.test import testconf
from au.test import testutils

import os

import numpy as np
import pytest

TEST_TEMPDIR = os.path.join(testconf.TEST_TEMPDIR_ROOT, 'test_mobilenet')

@pytest.mark.slow
def test_mobilenet_inference_graph(monkeypatch):
  testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)
  dataset.ImageTable.setup()

  params = mobilenet.Mobilenet.Small()
  model = mobilenet.Mobilenet.load_or_train(params)
  igraph = model.get_inference_graph()

  rows = dataset.ImageTable.iter_all_rows()
  filler = nnmodel.FillActivationsTFDataset(model=model)
  out_rows = list(filler(rows))

  # Test for smoke: logits for each image should be different ;)
  all_preds = set()
  for row in out_rows:
    acts = row.attrs['activations']
    assert model.igraph.model_name in acts.get_models()
    for tensor_name in model.igraph.output_names:
      t = acts.get_tensor(igraph.model_name, tensor_name)
      
      # Check that we have a non-empty array
      assert t.shape

      # If weights fail to load, the net will predict uniformly for
      # everything.  Make sure that doesn't happen!
      if tensor_name == 'MobilenetV2/Predictions/Reshape_1:0':
        preds = tuple(t)
        assert preds not in all_preds
        all_preds.add(preds)
      
        # The Small model consistently gets this one right
        uriss = ' '.join(row.uri for row in out_rows)
        assert '202228408_eccfe4790e.jpg' in uriss
        if '202228408_eccfe4790e.jpg' in row.uri:
          from slim.datasets import imagenet
          label_map = imagenet.create_readable_names_for_imagenet_labels()
          predicted = label_map[np.array(preds).argmax()]
          assert predicted == 'soccer ball'



    # For debugging, this is a Panda that the model predicts correctly per
    # https://colab.research.google.com/github/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_example.ipynb
    # import imageio
    # im = imageio.imread('https://upload.wikimedia.org/wikipedia/commons/f/fe/Giant_Panda_in_Beijing_Zoo_1.JPG')
    # import cv2
    # imr = cv2.resize(im, (96, 96))
    # print imr
    # y = self.endpoints['Predictions'].eval(feed_dict={input_image:[imr]})
    # print y, y.max()

def _test_mobilenet_activation_tables(monkeypatch, params_cls):
  testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)
  dataset.ImageTable.setup()
  
  with testutils.LocalSpark.sess() as spark:
    params = params_cls()

    class TestTable(nnmodel.ActivationsTable):
      TABLE_NAME = 'Mobilenet_test_' + params_cls.__name__
      NNMODEL_CLS = mobilenet.Mobilenet
      MODEL_PARAMS = params
      IMAGE_TABLE_CLS = dataset.ImageTable
  
    TestTable.setup(spark=spark)

@pytest.mark.slow
def test_mobilenet_activation_tables_small(monkeypatch):
  _test_mobilenet_activation_tables(monkeypatch, mobilenet.Mobilenet.Small)

@pytest.mark.slow
def test_mobilenet_activation_tables_medium(monkeypatch):
  _test_mobilenet_activation_tables(monkeypatch, mobilenet.Mobilenet.Medium)

@pytest.mark.slow
def test_mobilenet_activation_tables_large(monkeypatch):
  _test_mobilenet_activation_tables(monkeypatch, mobilenet.Mobilenet.Large)

@pytest.mark.slow
def test_mobilenet_activation_tables_xlarge(monkeypatch):
  _test_mobilenet_activation_tables(monkeypatch, mobilenet.Mobilenet.XLarge)