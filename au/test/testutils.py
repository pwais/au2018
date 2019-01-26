import itertools

from au import conf
from au import spark
from au.fixtures import dataset

class LocalSpark(spark.Spark):
  MASTER = 'local[8]'

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
    
    return open(temp_path).read()

