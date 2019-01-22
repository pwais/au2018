"""

Based upon tensorflow/models create_coco_tf_record.py and build_mscoco_data.py
https://github.com/tensorflow/models/blob/99256cf470df6af16808eb0e49a8354d2f9beae2/research/object_detection/dataset_tools/create_coco_tf_record.py
https://github.com/tensorflow/models/blob/848cc59225c1b44abdf3d3c42dc1c28723c0fab8/research/im2txt/im2txt/data/build_mscoco_data.py
https://github.com/tensorflow/models/blob/5444724e783fa7e517d54996553015dda994066e/research/object_detection/dataset_tools/download_and_preprocess_mscoco.sh

With differences:
 * For bounding box annotations, Tensoflow/models skips any bbox
    with an off-screen area.  In contrast, we do not omit such
    annotations, and instead provide a method for clamping the anno
    to image space.

"""

import os
import threading

from au import conf
from au import util
from au.fixtures import dataset
from au.spark import Spark

class BBox(object):
  __slots__ = (
    'x', 'y', 'width', 'height',
    'im_width', 'im_height',
    'area',
    'is_crowd',
    'category_id',
    'category_name',
    'anno_index', # To link with mask, if needed
  )

  def __getstate__(self):
    return self.to_dict()
  
  def __setstate__(self, d):
    for k in self.__slots__:
      setattr(self, k, d.get(k, ''))


  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  def to_dict(self):
    return dict((k, getattr(self, k, None)) for k in self.__slots__)

class Mask(object):
  __slots__ = (
    'png_bytes',
    'im_width', 'im_height',
    'is_crowd',
    'category_id',
    'category_name',
    'anno_index', # To link with bbox, if needed
  )

  def __getstate__(self):
    return self.to_dict()
  
  def __setstate__(self, d):
    for k in self.__slots__:
      setattr(self, k, d.get(k, ''))


  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  def to_dict(self):
    return dict((k, getattr(self, k, None)) for k in self.__slots__)

class Fixtures(object):

  BASE_ZIP_URL = "http://images.cocodataset.org/zips"
  BASE_ANNO_URL = "http://images.cocodataset.org/annotations"
  TRAIN_ZIP = "train2017.zip"
  TEST_ZIP = "test2017.zip"
  VAL_ZIP = "val2017.zip"
  ANNOS_TRAIN_VAL_ZIP = "annotations_trainval2017.zip"
  IMAGE_INFO_TEST_ZIP = "image_info_test2017.zip"

  ANNOS_TRAIN_FNAME = 'instances_train2017.json'
  ANNOS_VAL_FNAME = 'instances_val2017.json'

  DATA_ZIPS = (
    TRAIN_ZIP,
    TEST_ZIP,
    VAL_ZIP,
  )

  ANNO_ZIPS = (
    ANNOS_TRAIN_VAL_ZIP,
    IMAGE_INFO_TEST_ZIP,
  )

  ZIPS = tuple(list(DATA_ZIPS) + list(ANNO_ZIPS))

  ROOT = os.path.join(conf.AU_DATA_CACHE, 'mscoco')
  
  TEST_FIXTURE_DIR = os.path.join(conf.AU_DY_TEST_FIXTURES, 'mscoco')
  NUM_IMAGES_IN_TEST_ZIP = 100

  ## Source Data

  @classmethod
  def zips_dir(cls):
    return os.path.join(cls.ROOT, 'zips')

  @classmethod
  def zip_path(cls, fname):
    return os.path.join(cls.zips_dir(), fname)

  @classmethod
  def index_dir(cls):
    return os.path.join(cls.ROOT, 'index')

  ## Test Data

  @classmethod
  def test_fixture(cls, path):
    relpath = os.path.relpath(path, cls.ROOT)
    return os.path.join(cls.TEST_FIXTURE_DIR, relpath)

  ## Setup

  @classmethod
  def download_all(cls):
    for fname in cls.DATA_ZIPS:
      uri = cls.BASE_ZIP_URL + '/' + fname
      util.download(uri, cls.zip_path(fname), try_expand=False)
    for fname in cls.ANNO_ZIPS:
      uri = cls.BASE_ANNO_URL + '/' + fname
      util.download(uri, cls.zip_path(fname), try_expand=False)
  
  @classmethod
  def create_test_fixtures(cls):
    for fname in cls.ZIPS:
      src = cls.zip_path(fname)
      dest = cls.test_fixture(src)
      if fname in cls.DATA_ZIPS:
        util.copy_n_from_zip(src, dest, cls.NUM_IMAGES_IN_TEST_ZIP)
      elif fname in cls.ANNO_ZIPS:
        if not os.path.exists(dest):
          util.run_cmd('cp -v %s %s' % (src, dest))
  
  @classmethod
  def run_import(cls):
    print 'TODO make program'
    cls.download_all()
    cls.create_test_fixtures()

class AnnotationsIndexBase(object):
  FIXTURES = Fixtures
  ZIP_FNAME = ''
  ANNO_FNAME = ''

  _setup_lock = threading.Lock()
  _image_to_annos = None
  _image_id_to_image = None
  _category_index = None

  @classmethod
  def _index_file(cls, fname):
    index_dir = cls.__name__
    return os.path.join(cls.FIXTURES.index_dir(), index_dir, fname)

  @classmethod
  def setup(cls):
    with cls._setup_lock:
      cls._setup_indices()

  @classmethod
  def _setup_indices(cls):
    import shelve

    if not os.path.exists(cls._index_file('')):
      
      ###
      ### Based upon _create_tf_record_from_coco_annotations()
      ###
      
      import json
      import pprint

      # From tensorflow/models
      from object_detection.utils import label_map_util

      zip_path = cls.FIXTURES.zip_path(cls.ZIP_FNAME)
      util.log.info("Building annotations index for %s ..." % zip_path)

      fws = util.ArchiveFileFlyweight.fws_from(zip_path)
      anno_fw = None
      for fw in fws:
        if cls.ANNO_FNAME in fw.name:
          anno_fw = fw
      assert anno_fw, \
        "Could not find entry for %s in %s" % (cls.ANNO_FNAME, zip_path)

      util.log.info("... reading json ...")
      anno_data = json.loads(anno_fw.data)
      util.log.info("... json loaded ...")

      images = anno_data['images']
      category_index = label_map_util.create_category_index(
                                        anno_data['categories'])
      category_index = dict((str(k), v) for k, v in category_index.iteritems())
      
      util.log.info("Have annotations index for %s images." % len(images))
      util.log.info("Category index: \n\n%s" % pprint.pformat(category_index))

      image_to_annos = {}
      if 'annotations' in anno_data:
        util.log.info("... Building image ID -> Annos ...")
        for anno in anno_data['annotations']:
          # NB: we must string-ify keys for `shelve`
          image_id = str(anno['image_id'])
          image_to_annos.setdefault(image_id, [])
          image_to_annos[image_id].append(anno)

      missing_anno_count = sum(
        1 for image in images
        if str(image['id']) not in image_to_annos)
      util.log.info("... %s images are missing annos ..." % missing_anno_count)

      util.log.info("... finished index for %s ." % zip_path)

      image_id_to_image = dict((str(image['id']), image) for image in images)

      def dump_to_shelf(name, data):
        dest = cls._index_file(name)
        util.log.info("... saving %s to %s ..." % (name, dest))

        import pickle
        d = shelve.open(dest, protocol=pickle.HIGHEST_PROTOCOL)
        d.update(data.iteritems())
        d.close()

      # Keeping the below data in memory will OOM almost any reasonable box,
      # so we cache the data on disk.
      util.mkdir(cls._index_file(''))
      dump_to_shelf('image_id_to_image', image_id_to_image)
      dump_to_shelf('category_index', category_index)
      dump_to_shelf('image_to_annos', image_to_annos)

    util.log.info("Using indices in %s" % cls._index_file(''))
    cls._image_id_to_image = shelve.open(cls._index_file('image_id_to_image'))
    cls._category_index = shelve.open(cls._index_file('category_index'))
    cls._image_to_annos = shelve.open(cls._index_file('image_to_annos'))

  @classmethod
  def get_annos_for_image(cls, image_id):
    if not cls._image_to_annos:
      with cls._setup_lock:
        cls._setup_indices()
    return cls._image_to_annos.get(str(image_id), [])
  
  @classmethod
  def get_image_info(cls, image_id):
    if not cls._image_id_to_image:
      with cls._setup_lock:
        cls._setup_indices()
    return cls._image_id_to_image.get(str(image_id))
  
  @classmethod
  def get_category_name_for_id(cls, category_id):
    if not cls._category_index:
      with cls._setup_lock:
        cls._setup_indices()
    row = cls._category_index.get(str(category_id))
    if row:
      return row['name'].encode('utf8')
    else:
      return 'UNKNONW'.encode('utf8')
  
  @classmethod
  def get_bboxes_for_image(cls, image_id):
    image = cls.get_image_info(image_id)
    annos = cls.get_annos_for_image(image_id)
    
    bboxen = []
    if not (image and annos):
      util.log.warn("No annos or image info for image id %s" % image_id)
      return bboxen
    
    for anno_index, anno in enumerate(annos):
      kwargs = {
        'im_width': image['width'],
        'im_height': image['height'],
        'category_name': cls.get_category_name_for_id(anno['category_id']),
        'anno_index': anno_index,
      }
      kwargs.update(anno.iteritems())
      bbox = BBox(**anno)
      bboxen.append(bbox)
    return bboxen
  
  @classmethod
  def get_masks_for_image(cls, image_id):
    image = cls.get_image_info(image_id)
    annos = cls.get_annos_for_image(image_id)

    masks = []
    if not (image and annos):
      util.log.warn("No annos or image info for image id %s" % image_id)
      return masks

    for anno_index, anno in enumerate(annos):
      ## See Tensorflow/models create_coco_tf_record.py create_tf_example()
      from pycocotools import mask as cocomask
      import numpy as np
      import io
      import imageio
      run_len_encoding = cocomask.frPyObjects(
                                  anno['segmentation'],
                                  image['height'],
                                  image['width'])
      mask_arr = cocomask.decode(run_len_encoding)
      if not anno['iscrowd']:
        mask_arr = np.amax(mask_arr, axis=2) # hmmm ?
      buf = io.BytesIO()
      imageio.imwrite(buf, mask_arr, format='png')
      kwargs = {
        'png_bytes': buf.getvalue(),
        'im_width': image['height'],
        'im_height': image['width'],
        'category_name': cls.get_category_name_for_id(anno['category_id']),
        'anno_index': anno_index,
      }
      kwargs.update(anno.iteritems())
      mask = Mask(**kwargs)
      masks.append(mask)
    return masks

class TrainAnnos(AnnotationsIndexBase):
  ZIP_FNAME = Fixtures.ANNOS_TRAIN_VAL_ZIP
  ANNO_FNAME = Fixtures.ANNOS_TRAIN_FNAME

class ValAnnos(AnnotationsIndexBase):
  ZIP_FNAME = Fixtures.ANNOS_TRAIN_VAL_ZIP
  ANNO_FNAME = Fixtures.ANNOS_VAL_FNAME



class ImageURI(object):
  __slots__ = ('zip_path', 'image_fname')
  PREFIX = 'mscoco17://'

  def __init__(self, **kwargs):
    self.zip_path = kwargs.get('zip_path', '')
    self.image_fname = kwargs.get('image_fname', '')
  
  def to_str(self):
    return self.PREFIX + '|'.join((self.zip_path, self.image_fname))
  
  def __str__(self):
    return self.to_str()

  @staticmethod
  def from_uri(s):
    toks_s = s[len(ImageURI.PREFIX):]
    toks = toks_s.split('|')
    assert len(toks) == 2
    iu = ImageURI()
    iu.zip_path, iu.image_fname = toks
    return iu



class MSCOCOImageTableBase(dataset.ImageTable):

  FIXTURES = Fixtures

  ANNOS_CLS = None
  IMAGES_ZIP_FNAME = None

  # Annotation / Image Features to Include
  MASKS = True
  IMAGES = True
  BBOXEN = True

  # Pre-shuffle the data for SGD runs on the data in order (cache friendly)
  RANDOM_SHUFFLE = True
  RANDOM_SHUFFLE_SEED = 7
  APPROX_MB_PER_SHARD = 1024.
  # ROWS_PER_FILE ignored

  @classmethod
  def setup(cls, spark=None):
    spark = spark or Spark.getOrCreate()

    util.log.info("Setting up table %s (%s)" % (cls.TABLE_NAME, cls.__name__))

    cls.ANNOS_CLS.setup()

    split = ''
    if 'train' in cls.IMAGES_ZIP_FNAME:
      split = 'train'
    elif 'test' in cls.IMAGES_ZIP_FNAME:
      split = 'test'
    elif 'val' in cls.IMAGES_ZIP_FNAME:
      split = 'val'

    def gen_rows(fws):
      for fw in fws:
        if not ('.jpg' in fw.name or '.png' in fw.name):
          continue
        
        uri = ImageURI(zip_path=zip_path, image_fname=fw.name)
        image_bytes = ''
        if cls.IMAGES:
          image_bytes = fw.data
          assert len(image_bytes) > 0, 'Sanity check'

        fname = os.path.split(fw.name)[-1]
        image_id = int(fname.split('.')[0])
        info = cls.ANNOS_CLS.get_image_info(image_id)
        if info:
          attrs = {
            'width': info['width'],
            'height': info['height'],
            'mscoco_image_id': info['id'],
          }
        else:
          attrs = {}

        if cls.BBOXEN:
          attrs['mscoco_bboxen'] = cls.ANNOS_CLS.get_bboxes_for_image(image_id)
        
        if cls.MASKS:
          attrs['mscoco_masks'] = cls.ANNOS_CLS.get_masks_for_image(image_id)
        
        yield dataset.ImageRow(
          dataset='mscoco',
          split=split,
          uri=str(uri),
          image_bytes=image_bytes,
          attrs=attrs,
        )
    
    zip_path = cls.FIXTURES.zip_path(cls.IMAGES_ZIP_FNAME)
    fws = util.ArchiveFileFlyweight.fws_from(zip_path)
    if cls.RANDOM_SHUFFLE:
      import random
      g = random.Random()
      g.seed(cls.RANDOM_SHUFFLE_SEED)
      random.shuffle(fws, random=g.random)

    archive_rdd = spark.sparkContext.parallelize(fws, numSlices=len(fws))
    row_rdd = archive_rdd.mapPartitions(gen_rows)

    def estimate_bytes_per_row(rows):
      COMPRESSION_FRAC = 0.6
      import pickle
      n_rows = len(rows)
      assert all(r.image_bytes is not '' for r in rows), 'Sanity check'
      n_bytes = sum(len(pickle.dumps(r)) for r in rows)
      return COMPRESSION_FRAC * n_bytes / n_rows
    
    est_total_rows = archive_rdd.count()
    est_bytes_per_row = estimate_bytes_per_row(row_rdd.take(10))
    est_total_bytes = est_total_rows * est_bytes_per_row
    n_shards = max(10, int(est_total_bytes / (cls.APPROX_MB_PER_SHARD * 1e6)))

    util.log.info(
      "Writing %s total rows in %s shards; est. %s MB (%s MB / row)" % (
        est_total_rows,
        n_shards,
        est_total_bytes * 1e-6,
        est_bytes_per_row * 1e-6))

    # Partition the `archive_rdd` because partitioning `row_rdd` will cause
    # Spark to compute all ImageRows, which, as written above, are not
    # flyweights.
    row_rdd = archive_rdd.repartition(n_shards).mapPartitions(gen_rows)
    dataset.ImageRow.write_to_parquet(row_rdd, cls.table_root(), spark=spark)

class MSCOCOImageTableTrain(MSCOCOImageTableBase):
  TABLE_NAME = 'mscoco'
  ANNOS_CLS = TrainAnnos
  IMAGES_ZIP_FNAME = Fixtures.TRAIN_ZIP

class MSCOCOImageTableVal(MSCOCOImageTableBase):
  TABLE_NAME = 'mscoco'
  ANNOS_CLS = ValAnnos
  IMAGES_ZIP_FNAME = Fixtures.VAL_ZIP

