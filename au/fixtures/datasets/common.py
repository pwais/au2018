import copy

import numpy as np

from au.plotting import hash_to_rbg

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
      setattr(self, k, d.get(k, None))

  def __str__(self):
    return str(self.to_dict())

  def __init__(self, **kwargs):
    for k in self.__slots__:
      setattr(self, k, kwargs.get(k))
  
  def update(self, **kwargs):
    for k in self.__slots__:
      if k in kwargs:
        setattr(self, k, kwargs[k])

  def to_dict(self):
    return dict(
      (k, getattr(self, k, None))
      for k in self.__slots__)

  @staticmethod
  def of_size(width, height):
    return BBox(
            x=0, y=0,
            width=width, height=height,
            im_width=width, im_height=height)

  def set_x1_y1_x2_y2(self, x1, y1, x2, y2):
    self.update(x=x1, y=y1, width=x2 - x1 + 1, height=y2 - y1 + 1)

  def get_x1_y1_x2_y2(self):
    return self.x, self.y, self.x + self.width, self.y + self.height

  def get_x1_y1(self):
    return self.x, self.y

  def add_padding(self, *args):
    if len(args) == 1:
      px, py = args[0], args[0]
    elif len(args) == 2:
      px, py = args[0], args[1]
    else:
      raise ValueError(len(args))
    self.x -= px
    self.y -= py
    self.width += 2 * px
    self.height += 2 * py

  @staticmethod
  def from_x1_y1_x2_y2(x1, y1, x2, y2):
    b = BBox()
    b.set_x1_y1_x2_y2(x1, y1, x2, y2)
    return b

  def is_full_image(self):
    return (
      self.x == 0 and
      self.y == 0 and
      self.width == self.im_width and
      self.height == self.im_height)

  def get_corners(self):
    # From origin in CCW order
    return (
      (self.x, self.y),
      (self.x + self.width, self.y),
      (self.x + self.width, self.y + self.height),
      (self.x, self.y + self.height),
    )

  def get_num_onscreen_corners(self):
    return sum(
      1 for x, y in self.get_corners()
      if (0 <= x < self.im_width) and (0 <= y < self.im_height))

  def quantize(self):
    def quantize(v):
      return int(round(v)) if v is not None else v
    for attr in attrs:
      setattr(self, attr, quantize(getattr(self, attr)))

  def clamp_to_screen(self):
    def clip_and_norm(v, max_v):
      return int(np.clip(v, 0, max_v).round())
    
    x1, y1, x2, y2 = self.get_x1_y1_x2_y2()
    x1 = clip_and_norm(x1, self.im_width - 1)
    y1 = clip_and_norm(y1, self.im_height - 1)
    x2 = clip_and_norm(x2, self.im_width - 1)
    y2 = clip_and_norm(y2, self.im_height - 1)
    self.set_x1_y1_x2_y2(x1, y1, x2, y2)
    
  def get_intersection_with(self, other):
    ix1 = max(self.x, other.x)
    ix2 = min(self.x + self.width, other.x + other.width)
    iy1 = max(self.y, other.y)
    iy2 = min(self.y + self.height, other.y + other.height)
    
    intersection = copy.deepcopy(self)
    intersection.update(
      x=ix1, y=iy1, width=ix2-ix1, height=iy2-iy1)
    return intersection

  def get_union_with(self, other):
    ux1 = min(self.x, other.x)
    ux2 = max(self.x + self.width, other.x + other.width)
    uy1 = min(self.y, other.y)
    uy2 = max(self.y + self.height, other.y + other.height)
    
    union = copy.deepcopy(self)
    union.update(
      x=ux1, y=uy1, width=ux2-ux1, height=uy2-uy1)
    return union

  def overlaps_with(self, other):
    # TODO: faster
    return self.get_intersection_with(other).get_area() > 0

  def get_area(self):
    return self.width * self.height

  def translate(self, *args):
    if len(args) == 1:
      x, y = args[0].tolist()
    else:
      x, y = args
    self.x += x
    self.y += y

  def draw_in_image(self, img, color=None, thickness=2):
    assert self.im_height == img.shape[0], (self.im_height, img.shape)
    assert self.im_width == img.shape[1], (self.im_width, img.shape)

    if not color:
      color = hash_to_rbg(self.category_name)

    # Tensorflow takes BGR
    color = tuple(reversed(color))

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
        thickness=thickness,
        display_str_list=[self.category_name],
        use_normalized_coordinates=False)