from au import util

import os

import numpy as np

def hash_to_rbg(x, s=0.8, v=0.8):
  import colorsys
  import sys

  # NB: as of Python3, the seed to __hash__() is not stable,
  # but hash(int) is stable
  import hashlib
  h_i = int(hashlib.md5(str(x).encode('utf-8')).hexdigest(), 16)
  h = (hash(h_i) % 2654435769) / 2654435769.
  r, g, b = colorsys.hsv_to_rgb(h, s, v)
  
  return int(255. * r), int(255. * g), int(255. * b)

def img_to_data_uri(img, format='jpg'):
  """Given a numpy array `img`, return a `data:` URI suitable for use in 
  an HTML image tag."""

  from io import BytesIO
  out = BytesIO()

  import imageio
  imageio.imwrite(out, img, format=format)

  from base64 import b64encode
  data = b64encode(out.getvalue()).decode('ascii')
  
  from six.moves.urllib import parse
  data_url = 'data:image/png;base64,{}'.format(parse.quote(data))
  
  return data_url

def img_to_img_tag(img, display_scale=1, format='jpg'):
  h, w = img.shape[:2]
  h *= display_scale
  w *= display_scale
  src = img_to_data_uri(img, format=format)
  TEMPLATE = """<img src="{src}" height="{h}" width="{h}" />"""
  return TEMPLATE.format(src=src, h=h, w=w)

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
    ctxbox = Div(text=
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
