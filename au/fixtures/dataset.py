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
    
    'label',
#     '_label_bytes', # NB: see property label and label_bytes
#     '_cached_label',
#     '_cached_label_arr',
#     '_cached_label_fobj',
  )
  
  DEFAULT_PQ_PARTITION_COLS = ['dataset', 'split']
    # NB: must be a list and not a tuple due to c++ api
  
  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k, ''))
    
    if ('_image_bytes' not in kwargs and 
        kwargs.get('image_bytes', '') is not ''):
      self._image_bytes = kwargs['image_bytes']
    
#     if ('_label_bytes' not in kwargs and
#         kwargs.get('label_bytes', '') is not ''):
#       self._label_bytes = kwargs['label_bytes']
  
  @staticmethod
  def from_np_img_labels(np_img, label='', **kwargs):
    row = ImageRow(**kwargs)
    row._cached_image_arr = np_img
    row.label = label
    return row
  
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
#       elif k == '_label_bytes':
#         attrs.append(('label_bytes', self.label_bytes))
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

#   @property
#   def label_bytes(self):
#     if self._label_bytes is '':
#       # Read lazily
#       if self._cached_label_arr is not '':
#         buf = io.BytesIO()
#         imageio.imwrite(buf, self._cached_image_arr, format='png')
#         self._image_bytes = buf.getvalue()
#       elif self._cached_image_fobj is not '':
#         self._image_bytes = self._cached_image_fobj.read()
#         self._cached_image_fobj = ''
#     return self._image_bytes
# 
#   @property
#   def label(self):
#     if self._cached_label is '':
#       if self.label_encoding == 'json':
#         
#     
#     if self._label is '':
#       # Read lazily
#       if self._cached_label_arr is not '':
#         buf = io.BytesIO()
#         imageio.imwrite(buf, self._cached_label_arr, format='png')
#         self._label_bytes = buf.getvalue()
#       elif self._cached_label_fobj is not '':
#         self._label_bytes = self._cached_label_fobj.read()
#         self._cached_label_fobj = ''
#     return self._label_bytes

  def fname(self):
    has_fnamable_label = (
      self.label is not '' and
      isinstance(self.label, (basestring, int, float))) 
    
    toks = (
      self.dataset,
      self.split,
      'label_%s' % self.label if has_fnamable_label else '',
      self.uri.split('/')[-1] if self.uri else '',
    )
    
    fname = '-'.join(str(tok) for tok in toks if tok) + '.png'
    return fname

  def to_debug(self, fname=''):
    """Convenience for dumping an image to a place on disk where the user can
    view locally (e.g. using Apple Finder file preview, Ubuntu
    image browser, an nginx instance pointed at the folder, etc).
    
    FMI see conf.AU_CACHE_TMP
    """
    if self.image_bytes == '':
      return None
    
    dest = os.path.join(conf.AU_CACHE_TMP, self.fname())
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
  
  @staticmethod
  def from_pandas(df, **kwargs):
    for row in df.to_dict(orient='records'):
      row.update(**kwargs)
      yield ImageRow(**row)

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

  @staticmethod
  def write_to_pngs(rows, dest_root=None):
    log = create_log()
    
    dest_root = dest_root or conf.AU_DATA_CACHE
    
    log.info("Writing PNGs to %s ..." % dest_root)
    n = 0
    for row in rows:
      dest_dir = os.path.join(
                    dest_root,
                    row.dataset or 'default_dataset',
                    row.split or 'default_split')
      util.mkdir(dest_dir)
      
      fname = row.fname()
      
      dest = os.path.join(dest_dir, fname)
      with open(dest, 'wb') as f:
        f.write(row.image_bytes)
      
      n += 1
      if n % 100 == 0:
        log.info("... write %s PNGs ..." % n)
    log.info("... wrote %s total PNGs to %s ." % (n, dest_root))  
      
class ImageTable(object):
  """A table of images (can handle multiple datasets;  perhaps use one table
  per family of experiments).  Persisted as parquet."""
  
  TABLE_NAME = 'default'
  ROWS_PER_FILE = 100
  
  @classmethod
  def init(cls):
    """Subclasses should override to create a dataset from scratch
    (e.g. download images, create a table, etc)
    """
    pass
  
  @classmethod
  def table_root(cls):
    return os.path.join(conf.AU_TABLE_CACHE, cls.TABLE_NAME)
  
  @classmethod
  def save_to_image_table(cls, rows):
    dest = os.path.join(conf.AU_TABLE_CACHE, cls.TABLE_NAME)
    if not os.path.exists(dest):
      return ImageRow.write_to_parquet(
                        rows,
                        cls.table_root(),
                        rows_per_file=cls.ROWS_PER_FILE)

  @classmethod
  def get_rows_by_uris(cls, uris):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    
    pa_table = pq.read_table(cls.table_root())
    df = pa_table.to_pandas()
    matching = df[df.uri.isin(uris)]
    return list(ImageRow.from_pandas(matching))
    
#   @classmethod
#   def show_stats(cls, spark=None):
#     

#   @staticmethod
#   def write_tf_dataset_to_parquet(
#         dataset,
#         dest_dir,
#         

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