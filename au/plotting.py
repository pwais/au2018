from au import util

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

def histogram(spark_df, col, bins):
  """Compute and return a histogram of `bins` of the values in the column
  named `col` in spark Dataframe `spark_df`.  Return type is designed
  to match `numpy.histogram()`.
  """
  col_val_rdd = spark_df.select(col).map(unpack_pyspark_row)
  
  hist = np.zeros(len(bins))
  edges = np.zeros(len(bins) + 1)
  for i, interval in enumerate(col_val_rdd.histogram(bins)):
    if len(interval) == 3:
      lo, hi, count = interval
    elif len(interval) == 2:
      lo, count = interval
    hist[i] = count
    edges[i] = lo
  edges[-1] = col_val_rdd.max()
  return hist, edges

def save_bokeh_fig(fig, dest):
  from bokeh import plotting
  plotting.output_file(dest, title=fig.title, mode='inline')
  plotting.save(fig)
  util.log.info("Wrote to %s" % dest)

class HistogramWithExamplesPlotter(object):
  """Create and return a Bokeh plot depicting a histogram of a single column in
  a Spark DataFrame.  Clicking on a bar in the histogram will interactively
  show examples from that bucket.  Optionally facet the histogram using a
  second column (e.g. a category column).
  """

  NUM_BINS = 100

  MICRO_FACET_COL = None

  # Plotting params
  TITLE = None  # By default use DataFrame Column name

  def display_bucket(self, micro_facet, bucket_id, irows):
    import itertools
    rows_str = "<br />".join(str(r) for r in itertools.islice(irows, 5))
    TEMPLATE = """
      <b>Facet: {mf} Bucket: {bucket_id} </b> <br/>
      {rows}
      <br/> <br/>
    """
    return TEMPLATE.format(mf=micro_facet, bucket_id=bucket_id, rows=rows_str)

  def _build_data_source_for_micro_facet(self, mfv, df):
    util.log.info("... building data source for %s ..." % mfv)

    if mfv == 'ALL':
      mf_src_df = df
    else:
      mf_src_df = df.filter(df[self.MICRO_FACET_COL] == mfv)
      
    util.log.info("... histogramming %s ..." % mfv)
    hist, edges = histogram(df, col, self.NUM_BINS)

    # Use this Pandas Dataframe to serve as a bokeh data source
    # for the plot
    mf_df = pd.DataFrame(dict(
      count=hist, proportion=hist / np.sum(hist),
      left=edges[:-1], right=edges[1:],
    ))
    mf_df['legend'] = mfv

    from bokeh.colors import RGB
    mf_df['color'] = RGB(*hash_to_rbg(mf))
    
    util.log.info("... display-ifying examples for %s ..." % mfv)
    def get_display():
      from pyspark.sql import functions as F
      col_def = None
      buckets = list(zip(edges[:-1], edges[1:]))
      for bucket_id, (lo, hi) in enumerate(buckets):
        args = (
              (mf_src_df[metric] >= lo) & (mf_src_df[metric] < hi),
              bucket_id
        )
        if col_def is None:
          col_def = F.when(*args)
        else:
          col_def = col_def.when(*args)
      col_def = col_def.otherwise(-1)
      df_bucketed = mf_src_df.withColumn('ag_plot_bucket', col_def)
      bucketed_chunks = df_bucketed.rdd.groupBy(lambda r: r.ag_plot_bucket)
      bucket_to_disp = dict(
        bucketed_chunks.map(self.display_bucket).collect())
      return [
        bucket_to_disp.get(b, '')
        for b in range(len(buckets))
      ]
    mf_df['display'] = get_display()
    return mf_df

  def run(self, df, col):
    import pyspark.sql
    assert isinstance(df, pyspark.sql.DataFrame)
    assert col in df.columns
    
    util.log.info("Plotting histogram for %s of %s ..." % (col, df))
    
    micro_facet_values = ['ALL']
    if self.MICRO_FACET_COL:
      distinct_rows = df.select(self.MICRO_FACET_COL).distinct()

      micro_facet_values.extend(
        distinct_rows.rdd.map(unpack_pyspark_row).collect()
      )
    
    ## Compute a data source Pandas Dataframe for every micro-facet
    mfv_to_panel_df = {}
    for mfv in micro_facet_values:
      mfv_to_panel_df[mfv] = self._build_data_source_for_micro_facet(mfv, df)
    
    ## Make the plot
    fig = plotting.figure(
            title=self.TITLE or col,
            tools='tap,pan,wheel_zoom,box_zoom,reset',
            sizing_mode='scale_width',
            x_axis_label=col,
            y_axis_label='Count')
    for _, plot_src in legend_to_panel_df.items():
      plot_src = ColumnDataSource(plot_src)
      r = fig.quad(
        source=plot_src, bottom=0, top='count', left='left', right='right',
        color='color', fill_alpha=0.5,
        hover_fill_color='color', hover_fill_alpha=1.0,
        legend='legend')
      from bokeh.models import HoverTool
      fig.add_tools(
        HoverTool(
          # renderers=[r], FIXME ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          mode='vline',
          tooltips=[
            ('Facet', '@legend'),
            ('Count', '@count'),
            ('Proportion', '@proportion'),
            ('Value', '@left'),
          ]))

      fig.legend.click_policy = 'hide'

    ## Add the 'show examples' tool and div
    from bokeh.models.widgets import Div
    ctxbox = Div(text="Click on a histogram bar to show examples")

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
