import itertools

from au import conf
from au import spark
from au.fixtures import dataset

class LocalSpark(spark.Spark):
  MASTER = 'local[8]'

def create_video(n=40, w=1280, h=720, format='mov', fps=29.97):
  images_dir = conf.AU_IMAGENET_SAMPLE_IMGS_DIR
  rows = dataset.ImageRow.rows_from_images_dir(images_dir)
  imgs = [r.as_numpy() for r in rows]

  # Resize
  import cv2
  imgs = [cv2.resize(img, (w, h)) for img in imgs]

  # Imageio / ffmpeg must write to disk :/
  import tempfile
  temp_path = tempfile.NamedTemporaryFile().name + '.' + format

  import imageio
  writer = imageio.get_writer(temp_path, format=format, fps=fps)
  iter_imgs = itertools.islice(itertools.cycle(imgs), n)
  for img in iter_imgs:
    writer.append_data(img)
  writer.close()

  return open(temp_path).read()

