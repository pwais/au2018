import os

from au import conf
from au import spark
from au import util


class BDD100K(object):

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'bdd100k')

  ZIPS = os.path.join(ROOT, 'zips')

  TELEMETRY = os.path.join(ZIPS, 'bdd100k_info.zip')

  VIDEO_DIR = os.path.join(ROOT, 'videos')

  @classmethod
  def setup(cls, spark=None):

    ### Transform telemetry into Parquet table

    class Meta(object):
      __SLOTS__ = (
        'start_time',
        'end_time',
        'id',
        'filename',
        'timelapse',
        'ride_id',
      )

    class GPSObs(object):
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

    class Point3(object):
      __SLOTS__ = ('x', 'y', 'z')
      def __init__(self, **kwargs):
        self.x = kwargs.get('x')
        self.y = kwargs.get('y')
        self.z = kwargs.get('z')
    
    class Row(object):
      __SLOTS__ = (
        'namespace',
        'timestamp',
        'meta',
        
        'accelerometer',
        'gyro',
        'location',
        'gps',
      )
    
