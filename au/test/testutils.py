import itertools
import multiprocessing

from au import conf
from au import spark
from au.fixtures import dataset

class LocalSpark(spark.Spark):
  MASTER = 'local[%s]' % multiprocessing.cpu_count()

def to_png_bytes(arr):
  """When comparing images, we need to compare actual and expected via image
  bytes b/c imageio does some sort of subtle color normalization and we want
  our fixtures to simply be user-readable PNGs."""

  import io
  import imageio
  buf = io.BytesIO()
  imageio.imwrite(buf, arr, 'png')
  return buf.getvalue()

def iter_video_images(n, w, h):
  images_dir = conf.AU_IMAGENET_SAMPLE_IMGS_DIR
  rows = dataset.ImageRow.rows_from_images_dir(images_dir)
  imgs = [r.as_numpy() for r in rows]

  # Resize
  import cv2
  imgs = [cv2.resize(img, (w, h)) for img in imgs]
  
  # Yield `n`
  iimgs = itertools.islice(itertools.cycle(imgs), n)
  return iimgs

class VideoFixture(object):
  def __init__(self, **kwargs):
    self.n = kwargs.get('n', 30)
    self.w = kwargs.get('w', 32)
    self.h = kwargs.get('w', 32)
    self.format = kwargs.get('format', 'mov')
    self.fps = kwargs.get('fps', 10)
    self.codec = kwargs.get('codec', 'png') # Lossless; default is libx264
    self.imgs = kwargs.get('imgs', [])

  def get_bytes(self):
    # Imageio / ffmpeg must write to disk :/
    import tempfile
    temp_path = tempfile.NamedTemporaryFile().name + '.' + self.format

    if self.imgs:
      iimgs = iter(self.imgs)
    else:
      iimgs = iter_video_images(self.n, self.w, self.h)

    import imageio
    imageio.mimwrite(
      temp_path,
      iimgs,
      format=self.format, 
      fps=self.fps,
      codec=self.codec)
    
    return open(temp_path, 'rb').read()

