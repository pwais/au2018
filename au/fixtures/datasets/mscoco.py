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

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  def to_dict(self):
    return dict((k, getattr(self, k, None)) for k in self.__slots__)

class Fixtures(object):

  BASE_ZIP_URL = "http://images.cocodataset.org/zips"
  TRAIN_ZIP = "train2017.zip"
  TEST_ZIP = "test2017.zip"
  VAL_ZIP = "val2017.zip"
  ANNOS_TRAIN_VAL_ZIP = "annotations_trainval2017.zip"
  IMAGE_INFO_TEST_ZIP = "image_info_test2017.zip"

  ANNOS_TRAIN_FNAME = 'instances_train2017.json'
  ANNOS_VAL_FNAME = 'instances_train2017.json'

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

  ## Source Data

  @classmethod
  def zips_dir(cls):
    return os.path.join(cls.ROOT, 'zips')

  @classmethod
  def zip_path(cls, fname):
    return os.path.join(cls.zips_dir(), fname)

  ## Test Data

  @classmethod
  def test_fixture(cls, path):
    relpath = os.path.relpath(path, cls.ROOT)
    return os.path.join(cls.TEST_FIXTURE_DIR, relpath)

  ## Setup

  @classmethod
  def download_all(cls):
    for fname in cls.ZIPS:
      uri = cls.BASE_ZIP_URL + '/' + fname
      util.download(uri, cls.zip_path(fname), try_expand=False)
  
  @classmethod
  def create_test_fixtures(cls):
    for fname in cls.ZIPS:
      src = cls.zip_path(fname)
      dest = cls.test_fixture(src)
      if fname in cls.DATA_ZIPS:
        util.copy_n_from_zip(src, dest, 100)
      elif fname in cls.ANNO_ZIPS:
        util.run_cmd('cp -v %s %s' % (src dest))
  
class AnnotationsIndexBase(object):
  ZIP_PATH = ''
  ANNO_FNAME = ''

  _setup_lock = threading.Lock()
  _image_to_annos = {}
  _image_id_to_image = {}
  _category_index = None

  @classmethod
  def _setup_indices(cls):
    ###
    ### Based upon _create_tf_record_from_coco_annotations()
    ###
    
    import json
    import pprint

    # From tensorflow/models
    from object_detection.utils import label_map_util

    util.log.info("Building annotations index for %s ..." % cls.ZIP_PATH)

    fws = util.ArchiveFileFlyweight.fws_from(cls.ZIP_PATH)
    anno_fw = None
    for fw in fws:
      if cls.ANNO_FNAME in fw.name:
        anno_fw = fw
    assert anno_fw, \
      "Could not find entry for %s in %s" % (cls.ANNO_FNAME, cls.ZIP_PATH)

    anno_data = json.load(anno_fw.data)

    images = anno_data['images']
    cls._category_index = label_map_util.create_category_index(
                                      anno_data['categories'])
    
    util.log.info("Have annotations index for %s images." % len(images))
    util.log.info("Category index: \n\n%s" % pprint.pformat(category_index))

    cls._image_to_annos = {}
    if 'annotations' in anno_data:
      util.log.info("... Building image ID -> Annos ...")
      for anno in anno_data['annotations']:
        image_id = anno['image_id']
        image_to_annos.setdefault(image_id, [])
        image_to_annos[image_id].append(anno)
    
    missing_anno_count = sum(
      1 for image in images
      if image['id'] not in image_to_annos)
    util.log.info("... %s images are missing annos ..." % missing_anno_count)

    util.log.info("... finished index for %s ." % ZIP_PATH)
    cls._image_to_annos = image_to_annos
    cls._image_id_to_image = dict((image['id'], image) for image in images)

  @classmethod
  def get_annos_for_image(cls, image_id):
    if not cls._image_to_annos:
      with cls._setup_lock:
        cls._setup_indices()
    return cls._image_to_annos.get(image_id, [])
  
  @classmethod
  def get_image_info(cls, image_id):
    if not cls._image_id_to_image:
      with cls._setup_lock:
        cls._setup_indices()
    return cls._image_id_to_image.get(image_id)
  
  @classmethod
  def get_category_name_for_id(cls, category_id):
    if not cls._category_index:
      with cls._setup_lock:
        cls._setup_indices()
    row = cls._category_index.get(category_id)
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
      extra = {
        'im_width': image['width'],
        'im_height': image['height'],
        'category_name': cls.get_category_name_for_id(anno['category_id']),
        'anno_index': anno_index,
      }
      anno.update(**extra)
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
      return bboxen

    for anno_index, anno in enumeraet(annos):
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
        mask_arr = np.amax(binary_mask, axis=2)
      buf = io.BytesIO()
      imageio.imwrite(buf, mask_arr, format='png')
      kwargs = {
        'png_bytes': buf.getvalue(),
        'im_width': image['height'],
        'im_height': image['width'],
        'category_name': cls.get_category_name_for_id(anno['category_id']),
        'anno_index': anno_index,
      }
      kwargs.update(**anno)
      mask = Mask(**kwargs)
      masks.append(mask)
    return masks

class TrainAnnos(AnnotationsIndexBase):
  ZIP_PATH = FIXTURES.zip_path(FIXTURES.ANNOS_TRAIN_VAL_ZIP)
  ANNO_FNAME = FIXTURES.ANNOS_TRAIN_FNAME

class ValAnnos(AnnotationsIndexBase):
  ZIP_PATH = FIXTURES.zip_path(FIXTURES.ANNOS_TRAIN_VAL_ZIP)
  ANNO_FNAME = FIXTURES.ANNOS_VAL_FNAME

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

  @classmethod
  def setup(cls, spark=None):
    spark = spark or Spark.getOrCreate()

    split = ''
    if 'train' in cls.IMAGES_ZIP_FNAME:
      split = 'train'
    elif 'test' in cls.IMAGES_ZIP_FNAME:
      split = 'test'
    elif 'val' in cls.IMAGES_ZIP_FNAME:
      split = 'val'

    def iter_image_rows(fws):
      for fw in fws:
        if '.jpg' not in fw.name or '.png' not in fw.name:
          continue
        
        uri = ImageURI(zip_path=zip_path, image_fname=fw.name)
        image_bytes = ''
        if cls.IMAGES:
          image_bytes = fw.data

        image_id = int(fw.name.splt('.')[0])
        info = cls.ANNOS_CLS.get_image_info(image_id)
        attrs = {
          'width': info['width'],
          'height': info['height'],
          'mscoco_image_id': image['id'],
        }

        if cls.BBOXEN:
          attrs['mscoco_bboxen'] = cls.ANNOS_CLS.get_bboxes_for_image(image_id)
        
        if cls.MASKS:
          attrs['mscoco_masks'] = cls.ANNOS_CLS.get_masks_for_image(image_id)
        
        yield dataset.ImageRow(
          dataset='mscoco',
          split=split,
          uri=str(uri),
          _image_bytes=image_bytes,
          attrs=attrs,
        )
    
    zip_path = cls.FIXTURES.zip_path(cls.IMAGES_ZIP_FNAME)
    archive_rdd = Spark.archive_rdd(zip_path)
    row_rdd = archive_rdd.mapPartitions(iter_image_rows)

      

    
      
      