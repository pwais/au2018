from au import util
from au.fixtures import dataset
from au.fixtures import nnmodel
from au.fixtures.tf import mobilenet
from au.test import testconf

import os

import numpy as np
import pytest

# MODEL_BASEDIR = '/tmp/au_test/test_mobilenet'

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
    tensor_to_value = acts[0].tensor_to_value
    for tensor_name in model.igraph.output_names:
      assert tensor_name in tensor_to_value
      
      # Check that we have a non-empty array
      assert tensor_to_value[tensor_name].shape

      # If weights fail to load, the net will predict uniformly for
      # everything.  Make sure that doesn't happen!
      if tensor_name == 'MobilenetV2/Predictions/Reshape_1:0':
        preds = tuple(tensor_to_value[tensor_name])
        assert preds not in all_preds
        all_preds.add(preds)
      
        # The Small model consistently gets this one right
        if '202228408_eccfe4790e.jpg' in row.uri:
          from slim.datasets import imagenet
          label_map = imagenet.create_readable_names_for_imagenet_labels()
          predicted = label_map[np.array(preds).argmax()]
          assert predicted == 'soccer ball'





      #   print tensor_to_value[tensor_name], max(tensor_to_value[tensor_name])
      # # if tensor_name == 'MobilenetV2/embedding:0':
      # #   print tensor_to_value[tensor_name]#, max(tensor_to_value[tensor_name])
      
      # 
      # 


#   util.mkdir(MODEL_BASEDIR)
#   util.rm_rf(MODEL_BASEDIR)
  
#   params = mobilenet.Mobilenet.Small()
#   params.MODEL_BASEDIR = MODEL_BASEDIR
#   model = mobilenet.Mobilenet.load_or_train(params)
  
#   model2 = mobilenet.Mobilenet.load_or_train(params)
  
#   print list(model.iter_activations())
# #   label_map = imagenet.create_readable_names_for_imagenet_labels()  
# #   print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())