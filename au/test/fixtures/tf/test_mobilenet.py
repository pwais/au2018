from au import util
from au.fixtures import dataset
from au.fixtures import nnmodel
from au.fixtures.tf import mobilenet
from au.test import testconf

import os

import pytest

# MODEL_BASEDIR = '/tmp/au_test/test_mobilenet'

TEST_TEMPDIR = os.path.join(testconf.TEST_TEMPDIR_ROOT, 'test_mobilenet') 

@pytest.mark.slow
def test_mobilenet(monkeypatch):
  testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)
  dataset.ImageTable.setup()

  params = mobilenet.Mobilenet.Small()
  model = mobilenet.Mobilenet.load_or_train(params)
  igraph = model.get_inference_graph()
  assert igraph != nnmodel.TFInferenceGraphFactory()

  rows = list(dataset.ImageTable.iter_all_rows())
  assert rows

  filler = nnmodel.FillActivationsTFDataset(model=model)
  out_rows = list(filler(rows))
  assert len(out_rows) == len(rows)
  for row in out_rows:
    acts = row.attrs['activations']
    act = acts[0]
    assert act.model_name == igraph.model_name
    tensor_to_value = act.tensor_to_value
    for tensor_name in model.igraph.output_names:
      assert tensor_name in tensor_to_value
      
      # Check that we have a non-empty array
      assert tensor_to_value[tensor_name].shape


#   util.mkdir(MODEL_BASEDIR)
#   util.rm_rf(MODEL_BASEDIR)
  
#   params = mobilenet.Mobilenet.Small()
#   params.MODEL_BASEDIR = MODEL_BASEDIR
#   model = mobilenet.Mobilenet.load_or_train(params)
  
#   model2 = mobilenet.Mobilenet.load_or_train(params)
  
#   print list(model.iter_activations())
# #   label_map = imagenet.create_readable_names_for_imagenet_labels()  
# #   print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())