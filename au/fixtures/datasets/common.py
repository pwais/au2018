
def hash_to_rbga(x, s=0.8, v=0.8):
  import colorsys
  import sys  

  h = hash(str(x)) / float(sys.maxint)
  r, g, b = colorsys.hsv_to_rgb(h, s, v)
  return 255. * r, 255. * g, 255. * b, 255.

class BBox(object):
  """An object in an image; in particular, an (ideally amodal) bounding box
  surrounding the object.  May include additional context."""
  __slots__ = (
    'x', 'y', 'width', 'height',
    'im_width', 'im_height',
    'category_name',
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
    return dict(
      (k, getattr(self, k, None))
      for k in self.__slots__)

  def draw_in_image(self, img):
    assert self.im_height == img.shape[0]
    assert self.im_width == img.shape[1]

    color = hash_to_rbg(self.category_name)

    # Use Tensorflow Models
    from object_detection.utils.visualization_utils import \
      draw_bounding_box_on_image_array
    
    draw_bounding_box_on_image_array(
        img,
        self.y,
        self.x,
        self.y + self.height,
        self.x + self.width,
        color=color,
        thickness=2,
        display_str_list=[self.category_name],
        use_normalized_coordinates=False)