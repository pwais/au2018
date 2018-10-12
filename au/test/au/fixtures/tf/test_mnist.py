from au.fixtures.tf import mnist

import pytest

@pytest.mark.slow
def test_mnist_train():
  params = mnist.MNistEager.Params()
  params.TRAIN_EPOCHS = 1
  params.LIMIT = 10
  model = mnist.MNistEager.load_or_train(params)
  
  print list(model.iter_activations())
  