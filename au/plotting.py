from au import util

import os

import numpy as np

def hash_to_rbg(x, s=0.8, v=0.8):
  import colorsys
  import sys

  # NB: ideally we just use __hash__(), but as of Python 3 it's not stable
  import hashlib
  h_i = int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16)
  h = (hash(h_i) % 2654435769) / 2654435769.
  rgb = 255 * np.array(colorsys.hsv_to_rgb(h, s, v))
  
  return rgb.astype(int).tolist()

def color_to_opencv(color):
  r, g, b = np.clip(color, 0, 255).astype(int).tolist()
  return b, g, r

def draw_cuboid_xy_in_image(img, pts, base_color_rgb, alpha=0.3, thickness=2):
  """Given an image `img` and an array of n-by-(x,y) `pts` to be plotted in
  (r, g, b) color `base_color_rgb`, draw a cuboid in `img`.  We interpret
  the first four points of `pts` as the front of the cuboid (front face
  of object in it's ego frame), and the last four points as the back.
  We'll color the front face lighter than the rest of the cuboid edges."""

  base_color = np.array(base_color_rgb)
  front_color = color_to_opencv(base_color + 0.3 * 255)
  back_color = color_to_opencv(base_color - 0.3 * 255)
  center_color = color_to_opencv(base_color)

  import cv2
  # OpenCV can't draw transparent colors, so we use the 'overlay image' trick
  overlay = img.copy()

  front = pts[:4].astype(int)
  cv2.polylines(
    overlay,
    [front],
    True, # is_closed
    front_color,
    thickness)

  back = pts[4:, :2].astype(int)
  cv2.polylines(
    overlay,
    [back],
    True, # is_closed
    back_color,
    thickness)
  
  for start, end in zip(front.tolist(), back.tolist()):
    cv2.line(overlay, tuple(start), tuple(end), center_color, thickness)

  # Now blend!
  img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def draw_xy_depth_in_image(img, pts, dot_size=1, alpha=.2):
  """Given an image `img` and a point cloud `pts` [in form
  (pixel x, pixel y, ego depth meters)], draw the points.
  Point color varies every few meters with interpolation.
  """

  def rgb_for_distance(d_meters):
    DISTANCE_COLOR_PERIOD_METERS = 10
    bucket_below = int(d_meters / DISTANCE_COLOR_PERIOD_METERS)
    bucket_above = bucket_below + 1
    
    color_below = np.array(hash_to_rbg(bucket_below))
    color_above = np.array(hash_to_rbg(bucket_above))
    
    # Interpolate color to nearest bucket
    buckets = np.array([bucket_below, bucket_above])
    dist_above_below = buckets * DISTANCE_COLOR_PERIOD_METERS
    norm_dist_from_bucket = np.abs(d_meters - dist_above_below)
    weight = 1. - norm_dist_from_bucket / DISTANCE_COLOR_PERIOD_METERS
    w_below, w_above = weight.tolist()

    return w_below * color_below + w_above * color_above

  import cv2

  # OpenCV can't draw transparent colors, so we use the 'overlay image' trick:
  # First draw dots an an overlay...
  overlay = img.copy()
  for x, y, d_meters in pts.tolist():
    color = color_to_opencv(rgb_for_distance(d_meters))
    x = int(round(x))
    y = int(round(y))
    cv2.circle(overlay, (x, y), dot_size, color, thickness=2)
  
  # Now blend!
  img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def img_to_data_uri(img, format='jpg', jpeg_quality=75):
  """Given a numpy array `img`, return a `data:` URI suitable for use in 
  an HTML image tag."""

  from io import BytesIO
  out = BytesIO()

  import imageio
  kwargs = dict(format=format)
  if format == 'jpg':
    kwargs.update(quality=jpeg_quality)
  imageio.imwrite(out, img, **kwargs)

  from base64 import b64encode
  data = b64encode(out.getvalue()).decode('ascii')
  
  from six.moves.urllib import parse
  data_url = 'data:image/png;base64,{}'.format(parse.quote(data))
  
  return data_url

def get_hw_in_viewport(img_hw, viewport_hw):
  vh, vw = viewport_hw
  h, w = img_hw
  if h > vh:
    rescale = float(vh) / h
    h = rescale * h
    w = rescale * w
  if w > vw:
    rescale = float(vw) / w
    h = rescale * h
    w = rescale * w
  return int(h), int(w)

def img_to_img_tag(
    img,
    display_viewport_hw=None, # E.g. (1000, 1000)
    image_viewport_hw=None,   # E.g. (1000, 1000)
    format='jpg',
    jpeg_quality=75):

  if image_viewport_hw is not None:
    th, tw = get_hw_in_viewport(img.shape[:2], image_viewport_hw)
    import cv2
    img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_NEAREST)
  
  dh, dw = img.shape[:2]
  if display_viewport_hw is not None:
    dh, dw = get_hw_in_viewport((dh, dw), display_viewport_hw)

  src = img_to_data_uri(img, format=format, jpeg_quality=jpeg_quality)
  TEMPLATE = """<img src="{src}" height="{dh}" width="{dw}" />"""
  return TEMPLATE.format(src=src, dh=dh, dw=dw)

def unpack_pyspark_row(r):
  """Unpack a `pyspark.sql.Row` that contains a single value."""
  return r[0]
    # NB: pyspark.sql.Row is indexable

def df_histogram(spark_df, col, num_bins):
  """Compute and return a histogram of `bins` of the values in the column
  named `col` in spark Dataframe `spark_df`.  Return type is designed
  to match `numpy.histogram()`.
  """
  assert num_bins >= 1
  
  col_val_rdd = spark_df.select(col).rdd.map(unpack_pyspark_row)
  
  buckets, counts = col_val_rdd.histogram(num_bins)
  return np.array(counts), np.array(buckets)

def save_bokeh_fig(fig, dest, title=None):
  from bokeh import plotting
  if not title:
    title = os.path.split(dest)[-1]
  plotting.output_file(dest, title=title, mode='inline')
  plotting.save(fig)
  util.log.info("Wrote to %s" % dest)

class HistogramWithExamplesPlotter(object):
  """Create and return a Bokeh plot depicting a histogram of a single column in
  a Spark DataFrame.  Clicking on a bar in the histogram will interactively
  show examples from that bucket.  Optionally facet the histogram using a
  second column (e.g. a category column).
  
  The user can override how examples are displayed; subclasses can override
  `HistogramWithExamplesPlotter::display_bucket()`

  See `HistogramWithExamplesPlotter::run()`.
  """

  ## Core Params
  NUM_BINS = 50

  SUB_PIVOT_COL = None

  WIDTH = 1000
    # Bokeh's plots (especially in single-column two-row layout we use) work
    # best with a fixed width

  ## Plotting params
  TITLE = None  # By default use DataFrame Column name

  def display_bucket(self, sub_pivot, bucket_id, irows):
    import itertools
    rows_str = "<br />".join(str(r) for r in itertools.islice(irows, 5))
    TEMPLATE = """
      <b>Facet: {spv} Bucket: {bucket_id} </b> <br/>
      {rows}
      <br/> <br/>
    """
    disp = TEMPLATE.format(spv=sub_pivot, bucket_id=bucket_id, rows=rows_str)
    return bucket_id, disp

  def _build_data_source_for_sub_pivot(self, spv, df, col):
    import pandas as pd
    util.log.info("... building data source for %s ..." % spv)

    if spv == 'ALL':
      sp_src_df = df
    else:
      sp_src_df = df.filter(df[self.SUB_PIVOT_COL] == spv)
      
    util.log.info("... histogramming %s ..." % spv)
    hist, edges = df_histogram(sp_src_df, col, self.NUM_BINS)

    # Use this Pandas Dataframe to serve as a bokeh data source
    # for the plot
    sp_df = pd.DataFrame(dict(
      count=hist, proportion=hist / np.sum(hist),
      left=edges[:-1], right=edges[1:],
    ))
    sp_df['legend'] = str(spv)

    from bokeh.colors import RGB
    sp_df['color'] = RGB(*hash_to_rbg(spv))
    
    util.log.info("... display-ifying examples for %s ..." % spv)
    def get_display():
      # First, we need to re-bucket each row using the buckets collected
      # via the `df_histogram()` call above.  We'll use a Spark
      # `when`-`otherwise` function to make this bucket mapping efficient
      # (i.e. optimized into native code at runtime).
      from pyspark.sql import functions as F
      col_def = None
      buckets = list(zip(edges[:-1], edges[1:]))
      for bucket_id, (lo, hi) in enumerate(buckets):
        # The last spark histogram bucket is closed, but we want open
        if bucket_id == len(buckets) - 1:
          hi += 1e-9
        args = (
          (sp_src_df[col] >= float(lo)) & (sp_src_df[col] < float(hi)),
          bucket_id
        )
        if col_def is None:
          col_def = F.when(*args)
        else:
          col_def = col_def.when(*args)
      col_def = col_def.otherwise(-1)
      df_bucketed = sp_src_df.withColumn('au_plot_bucket', col_def)
      
      # Second, we collect chunks of rows partitioned by bucket ID so that we
      # can run our display function in parallel over buckets.
      bucketed_chunks = df_bucketed.rdd.groupBy(lambda r: r.au_plot_bucket)
      bucket_disp = bucketed_chunks.map(
                      lambda b_irows: 
                        self.display_bucket(spv, b_irows[0], b_irows[1]))
      bucket_to_disp = dict(bucket_disp.collect())
      
      # Finally, return a column of display strings ordered by buckets so that
      # we can add this column to the output histogram DataFrame.
      return [
        bucket_to_disp.get(b, '')
        for b in range(len(buckets))
      ]
    sp_df['display'] = get_display()
    return sp_df

  def run(self, df, col):
    """Compute histograms and return the final plot.

    Args:
      df (pyspark.sql.DataFrame): Read from this DataFrame.  The caller may
        want to `cache()` the DataFrame as this routine will do a variety of
        random reads and aggregations on the data.
      col (str): The x-axis for the computed histogram shall this this column
        in `df` as the chosen metric.  Spark automatically ignores nulls and
        nans.

    Returns:
      bokeh layout object with a plot.
    """
    import pyspark.sql
    assert isinstance(df, pyspark.sql.DataFrame)
    assert col in df.columns
    
    util.log.info("Plotting histogram for %s of %s ..." % (col, df))
    
    sub_pivot_values = ['ALL']
    if self.SUB_PIVOT_COL:
      distinct_rows = df.select(self.SUB_PIVOT_COL).distinct()

      sub_pivot_values.extend(
        sorted(
          distinct_rows.rdd.map(unpack_pyspark_row).collect()))
    
    ## Compute a data source Pandas Dataframe for every micro-facet
    spv_to_panel_df = dict(
      (spv, self._build_data_source_for_sub_pivot(spv, df, col))
      for spv in sub_pivot_values)
    
    ## Make the plot
    from bokeh import plotting
    fig = plotting.figure(
            title=self.TITLE or col,
            tools='tap,pan,wheel_zoom,box_zoom,reset',
            width=self.WIDTH,
            x_axis_label=col,
            y_axis_label='Count')
    for spv in sub_pivot_values:
      plot_src = spv_to_panel_df[spv]
      from bokeh.models import ColumnDataSource
      plot_src = ColumnDataSource(plot_src)
      r = fig.quad(
        source=plot_src, bottom=0, top='count', left='left', right='right',
        color='color', fill_alpha=0.5,
        hover_fill_color='color', hover_fill_alpha=1.0,
        legend='legend')
      from bokeh.models import HoverTool
      fig.add_tools(
        HoverTool(
          renderers=[r],
            # For whatever reason, adding a hover tool for each quad
            # makes the interface dramatically faster in the browser
          mode='vline',
          tooltips=[
            ('Sub-pivot', '@legend'),
            ('Count', '@count'),
            ('Proportion', '@proportion'),
            ('Value of %s' % col, '@left'),
          ]))

      fig.legend.click_policy = 'hide'

    ## Add the 'show examples' tool and div
    from bokeh.models.widgets import Div
    ctxbox = Div(width=self.WIDTH, text=
        "Click on a histogram bar to show examples.  "
        "Click on the legend to show/hide a series.")


    from bokeh.models import TapTool
    taptool = fig.select(type=TapTool)

    from bokeh.models import CustomJS
    taptool.callback = CustomJS(
      args=dict(ctxbox=ctxbox),
      code="""
        var idx = cb_data.source.selected['1d'].indices[0];
        ctxbox.text='' + cb_data.source.data.display[idx];
      """)

    from bokeh.layouts import column
    layout = column(fig, ctxbox)
    return layout
