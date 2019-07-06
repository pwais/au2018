from au import util
from au.fixtures.datasets import argoverse as av
from au.test import testconf
from au.test import testutils

import os
import unittest

import pytest

class TestArgoverseImageTable(unittest.TestCase):
  """Exercise utilties in the Argoverse module.  Allow soft failures
  if the user has none of the required tarballs.  We assume exclusively
  one of:
     1) the user emplaced the fixtures correctly using aucli
     2) the user has no fixtures and does not need them
  """

  @classmethod
  def setUpClass(cls):
    tracking_sample = av.Fixtures.tarball_dir(av.Fixtures.TRACKING_SAMPLE)
    cls.have_fixtures = os.path.exists(tracking_sample)
    
    from _pytest.monkeypatch import MonkeyPatch
    monkeypatch = MonkeyPatch()
    TEST_TEMPDIR = os.path.join(
                        testconf.TEST_TEMPDIR_ROOT,
                        'test_argoverse')
    testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)

  def test_sample(self):
    if not self.have_fixtures:
      return
    
    test_uri = av.FrameURI(
                  tarball_name=av.Fixtures.TRACKING_SAMPLE,
                  log_id='c6911883-1843-3727-8eaa-41dc8cda8993')

    loader = av.Fixtures.get_loader(test_uri)
    print('Loaded', loader)
    assert loader.image_count == 3441


