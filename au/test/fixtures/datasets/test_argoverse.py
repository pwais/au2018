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
    
    # from _pytest.monkeypatch import MonkeyPatch
    # monkeypatch = MonkeyPatch()
    # TEST_TEMPDIR = os.path.join(
    #                     testconf.TEST_TEMPDIR_ROOT,
    #                     'test_argoverse')
    # testconf.use_tempdir(monkeypatch, TEST_TEMPDIR)

  def test_basic(self):
    assert av.Fixtures.TRACKING_SAMPLE in av.Fixtures.all_tarballs()
    assert av.Fixtures.TRACKING_SAMPLE in av.Fixtures.all_tracking_tarballs()

  def test_sample(self):
    if not self.have_fixtures:
      return
    
    test_uri = av.FrameURI(
                  tarball_name=av.Fixtures.TRACKING_SAMPLE,
                  log_id='c6911883-1843-3727-8eaa-41dc8cda8993')

    loader = av.Fixtures.get_loader(test_uri)
    print('Loaded', loader)
    assert loader.image_count == 3441

    all_uris = list(av.Fixtures.iter_image_uris('sample'))
    assert len(all_uris) == 3441
    
    EXPECTED_URI = 'argoverse://tarball_name=tracking_sample.tar.gz&log_id=c6911883-1843-3727-8eaa-41dc8cda8993&split=sample&camera=ring_front_center&timestamp=315978406365860408'
    assert EXPECTED_URI in set(str(uri) for uri in all_uris)

    frame = av.AVFrame(uri=EXPECTED_URI)
    import imageio
    # TODO create fixture
    imageio.imwrite('/opt/au/tasttt.png', frame.debug_image,format='png')

    with testutils.LocalSpark.sess() as spark:
      av.Fixtures.label_df(spark, splits=('sample',))



