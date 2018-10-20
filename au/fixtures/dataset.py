import io
import os
from collections import OrderedDict

import imageio
import numpy as np

from au import conf
from au.util import create_log
from au import util

class ImageRow(object):
  """For expected usage, see `test_imagerow_demo`"""
  
  __slots__ = (
    'dataset',
    'split',
    'uri',
    
    '_image_bytes',  # NB: see property image_bytes
    '_cached_image_arr', # TODO: use __ for privates .. idk 
    '_cached_image_fobj',
    
    'label_type',
    '_label_bytes',  # NB: see property label_bytes
    '_cached_label_arr',
    '_cached_label_fobj',
  )
  
  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k, ''))
    
    if ('_image_bytes' not in kwargs and 
        kwargs.get('image_bytes', '') is not ''):
      self.image_bytes = kwargs['image_bytes']
    
    if ('_label_bytes' not in kwargs and
        kwargs.get('label_bytes', '') is not ''):
      self.label_bytes = kwargs['label_bytes']
  
  @staticmethod
  def from_path(path, **kwargs):
    # NB: The ImageRow instance will be a flyweight for the image data
    row = ImageRow(uri=path, **kwargs)
    row._cached_image_fobj = open(path, 'rb')
    return row
  
  def to_dict(self):
    attrs = []
    for k in self.__slots__:
      if not k.startswith('_'):
        attrs.append((k, getattr(self, k)))
      elif k == '_image_bytes':
        attrs.append(('image_bytes', self.image_bytes))
      elif k == '_label_bytes':
        attrs.append(('label_bytes', self.label_bytes))
    return OrderedDict(attrs)
  
  def as_numpy(self):
    if self._cached_image_arr is '':
      image_bytes = self.image_bytes
      if image_bytes is '':
        # Can't make an array
        return np.array([])
      
      self._cached_image_arr = imageio.imread(io.BytesIO(image_bytes))
    return self._cached_image_arr
  
  @property
  def image_bytes(self):
    if self._image_bytes is '':
      # Read lazily
      if self._cached_image_arr is not '':
        buf = io.BytesIO()
        imageio.imwrite(buf, self._cached_image_arr, format='png')
        self._image_bytes = buf.getvalue()
      elif self._cached_image_fobj is not '':
        self._image_bytes = self._cached_image_fobj.read()
        self._cached_image_fobj = ''
    return self._image_bytes

  @property
  def label_bytes(self):
    if self._label_bytes is '':
      # Read lazily
      if self._cached_label_arr is not '':
        buf = io.BytesIO()
        imageio.imwrite(buf, self._cached_label_arr, format='png')
        self._label_bytes = buf.getvalue()
      elif self._cached_label_fobj is not '':
        self._label_bytes = self._cached_label_fobj.read()
        self._cached_label_fobj = ''
    return self._label_bytes

  def to_debug(self, fname=''):
    """Convenience for dumping an image to a place on disk where the user can
    view locally (e.g. using Apple Finder file preview, Ubuntu
    image browser, an nginx instance pointed at the folder, etc).
    
    FMI see conf.AU_CACHE_TMP
    """
    if self.image_bytes == '':
      return None
    
    toks = (self.dataset, self.split, self.uri.split('/')[-1])
    fname = fname or '-'.join(str(t) for t in toks if t)
    dest = os.path.join(conf.AU_CACHE_TMP, fname + '.png')
    util.mkdir(conf.AU_CACHE_TMP)
    with open(dest, 'wb') as f:
      f.write(self.image_bytes)
    return dest 

  @staticmethod
  def rows_from_images_dir(img_dir, pattern='*', **kwargs):
    import pathlib2 as pathlib
    
    log = create_log()
    
    log.info("Reading images from dir %s ..." % img_dir)
    paths = pathlib.Path(img_dir).glob(pattern)
    n = 0
    for path in paths:
      path = str(path) # pathlib uses PosixPath thingies ...
      yield ImageRow.from_path(path, **kwargs)
      
      n += 1
      if (n % 100) == 0:
        log.info("... read %s paths ..." % n)
    log.info("... read %s total paths." % n)
  
  
  
  DEFAULT_PQ_PARTITION_COLS = ['dataset', 'split']

  @staticmethod
  def write_to_parquet(
        rows,
        dest_dir,
        rows_per_file=-1,
        partition_cols=DEFAULT_PQ_PARTITION_COLS,
        compression='snappy'):
    
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    log = create_log()
    
    if rows_per_file >= 1:
      irows = util.ichunked(rows, rows_per_file)
    else:
      rows = list(rows)
      if not rows:
        return
      irows = iter([rows])
    
    log.info("Writing parquet to %s ..." % dest_dir)
    for row_chunk in irows:
      r = row_chunk[0]
      
      # Pandas wants dicts
      if isinstance(r, ImageRow):
        row_chunk = [r.to_dict() for r in row_chunk]
#       import ipdb; ipdb.set_trace()
      df = pd.DataFrame(row_chunk)
      table = pa.Table.from_pandas(df)
      util.mkdir(dest_dir)
      pq.write_to_dataset(
            table,
            dest_dir,
            partition_cols=partition_cols,
            preserve_index=False, # Don't care about pandas index
            compression=compression,
            flavor='spark')
      log.info("... wrote %s rows ..." % len(row_chunk))
    log.info("... done writing to %s ." % dest_dir)
    

"""
make a dataset for 1-channel mnist things

make a dataset for our handful of images

try to coerce dataset from mscoco

make one for bbd100k

record activations for mnist
then for mobilenet on bdd100k / mscoco
take note of deeplab inference: https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb#scrollTo=edGukUHXyymr
and we'll wanna add maskrcnn mebbe ?

SPARK_LOCAL_IP=127.0.0.1 $SPARK_HOME/bin/pyspark --packages databricks:tensorframes:0.5.0-s_2.11 --packages databricks:spark-deep-learning:1.2.0-spark2.3-s_2.11


"""


class DatasetFactoryBase(object):
  
  class ParamsBase(object):
    def __init__(self):
      self.BASE_DIR = ''
  
  @classmethod
  def create_dataset(cls):
    pass
  
  @classmethod
  def get_ctx_for_entry(cls, entry_id):
    pass