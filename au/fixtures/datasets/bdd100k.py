import os

from au import conf
from au import spark
from au import util

class MutableTuple(object):
  __SLOTS__ = tuple()

  def __init__(self, **kwargs):
    for k in self.__SLOTS__:
      setattr(self, k, kwargs.get(k))

class Meta(MutableTuple):
  __SLOTS__ = (
    'startTime',
    'endTime',
    'id',
    'filename',
    'timelapse',
    'rideID',
  )

class GPSObs(MutableTuple):
  __SLOTS__ = (
    'altitude', 
    'longitude', 
    'vertical accuracy', 
    'horizontal accuracy', 
    'latitude', 
    'speed',

    'accuracy',
    'course',
  )

class Point3(MutableTuple):
  __SLOTS__ = ('x', 'y', 'z')
  

class TimeseriesRow(object):
  __SLOTS__ = (
    'namespace',
    'timestamp',
    'meta',
    
    'accelerometer',
    'gyro',
    'location',
    'gps',
  )

def json_to_rows(jobj, namespace_prefix=''):
  meta = Meta(**jobj)
  namespace = namespace_prefix + '.' if namespace_prefix else ''
  namespace = 'bdd100k.' + meta.filename
  
  for datum in jobj.get('gps', []):
    yield TimeseriesRow(
      namespace=namespace,
      timestamp=datum['timestamp'],
      meta=meta,
      gps=GPSObs(**datum),
    )
  
  for datum in jobj.get('location', []):
    yield TimeseriesRow(
      namespace=namespace,
      timestamp=datum['timestamp'],
      meta=meta,
      location=GPSObs(**datum),
    )
  
  for datum in jobj.get('accelerometer', []):
    yield TimeseriesRow(
      namespace=namespace,
      timestamp=datum['timestamp'],
      meta=meta,
      accelerometer=Point3(**datum),
    )
  
  for datum in jobj.get('gyro', []):
    yield TimeseriesRow(
      namespace=namespace,
      timestamp=datum['timestamp'],
      meta=meta,
      gyro=Point3(**datum),
    )

class BDD100K(object):

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'bdd100k')

  ZIPS = os.path.join(ROOT, 'zips')

  TELEMETRY = os.path.join(ZIPS, 'bdd100k_info.zip')

  VIDEO_DIR = os.path.join(ROOT, 'videos')

  @classmethod
  def setup(cls, spark=None):

    ### Transform telemetry into Parquet table

    
    
