from au import util
from au.fixtures.tf import mobilenet

import pytest

MODEL_BASEDIR = '/tmp/au_test/test_mobilenet'

#@pytest.mark.slow
def test_mobilenet():
  util.mkdir(MODEL_BASEDIR)
  util.rm_rf(MODEL_BASEDIR)
  
  params = mobilenet.Mobilenet.Small()
  params.MODEL_BASEDIR = MODEL_BASEDIR
  model = mobilenet.Mobilenet.load_or_train(params)
  
  model2 = mobilenet.Mobilenet.load_or_train(params)
  
  print list(model.iter_activations())
#   label_map = imagenet.create_readable_names_for_imagenet_labels()  
#   print("Top 1 prediction: ", x.argmax(),label_map[x.argmax()], x.max())