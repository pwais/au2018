from au.fixtures import dataset
from au.fixtures import nnmodel
from au.fixtures.tf import alm
from au.test import testconf
from au.test import testutils

import os
import unittest

import numpy as np

import pytest

TEST_TEMPDIR = os.path.join(testconf.TEST_TEMPDIR_ROOT, 'alm_test')

class TestBasicAutoencoder(unittest.TestCase):

  @classmethod
  def setUpClass(cls):
    # Use /tmp for test fixtures
    from _pytest.monkeypatch import MonkeyPatch
    monkeypatch = MonkeyPatch()
    testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)
  
    dataset.ImageTable.setup()

  def test_example_xform(self):

    def filled_mock_activations(row):
      arr = row.as_numpy()
      acts = nnmodel.Activations()
      acts.set_tensor('visible', 'img', arr)
      row.attrs = {}
      row.attrs['activations'] = acts
      return row
    
    rows = [
      filled_mock_activations(r)
      for r in dataset.ImageTable.iter_all_rows()
    ]

    xform = alm.ImageRowToExampleXForm()
    for row in rows:
      ex = xform(row)
      np.testing.assert_array_equal(ex.x, ex.y.flatten())
      assert ex.uri == row.uri
    
