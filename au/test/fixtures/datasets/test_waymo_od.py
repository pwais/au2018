import os

from au import conf

def test_get_jpeg_size():
  from au.fixtures.datasets.waymo_od import get_jpeg_size
  for fname in ('1292397550_115450d9bc.jpg', '1345687_fde3a33c03.jpg'):
    path = os.path.join(conf.AU_IMAGENET_SAMPLE_IMGS_DIR, fname)
    jpeg_bytes = open(path, 'rb').read()
    width, height = get_jpeg_size(jpeg_bytes)

    import imageio
    expected_h, expected_w, expected_c = imageio.imread(path).shape
    assert (width, height) == (expected_w, expected_h)
