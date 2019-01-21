from au.fixtures.datasets import mscoco
from au.test import testconf
from au.test import testutils

import unittest

import pytest

class TestFixtures(mscoco.Fixtures):
  ROOT = mscoco.Fixtures.TEST_FIXTURE_DIR

class TestMSCOCOImageTable(mscoco.MSCOCOImageTable):
  FIXTURES = TestFixtures

class TestMSCOCOImageTable(unittest.TestCase):
  """Exercise utiltiies in the mscoco module.  Allow soft failures
  if the user has none of the required zip files.  We assume exclusively
  one of:
     1) the user emplaced the fixtures correctly using aucli
     2) the user has no fixtures and does not need them
  """

  @classmethod
  def setUpClass(cls):
    cls.have_fixtures = False
    try:
      mscoco.Fixtures.create_test_fixtures()
      cls.have_fixtures = True
    except Exception as e:
      print "Failed to create test fixtures: %s" % (e,)
  
  @pytest.mark.slow
  def test_image_table(self):
    if not self.have_fixtures:
      return
    
