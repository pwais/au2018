from au import conf
from au import plotting as aupl
from au.test import testconf
from au.test import testutils

import numpy as np
import pytest

import os

def test_hash_to_rgb():
  assert aupl.hash_to_rbg('moof') == (204, 40, 144)
  assert aupl.hash_to_rbg(5) == (204, 155, 40)
  assert aupl.hash_to_rbg('moof1') != aupl.hash_to_rbg('moof')

def test_draw_xy_depth_in_image():
  # Create points for a test image:
  #  * One point every 10 pixels in x- and y- directions
  #  * The depth value of the pixel is the scalar value of the y-coord
  #      interpreted as meters
  h, w = 600, 600
  pts = []
  for y in range(int(h / 10)):
    for x in range(int(w / 10)):
      pts.append((x * 10, y * 10, y))
  
  pts = np.array(pts)
  actual = np.zeros((h, w, 3))
  aupl.draw_xy_depth_in_image(actual, pts)

  FIXTURES_BASE_PATH = os.path.join(conf.AU_ROOT, 'au/test/')
  FIXTURE_NAME = 'test_draw_xy_depth_in_image.png'
  
  actual_bytes = testutils.to_png_bytes(actual)
  expected_bytes = \
    open(os.path.join(FIXTURES_BASE_PATH, FIXTURE_NAME), 'rb').read()

  assert actual_bytes == expected_bytes

@pytest.mark.slow
def test_spark_histogram():
  with testutils.LocalSpark.sess() as spark:
    from pyspark.sql import Row
    df = spark.createDataFrame([Row(a=a, b=a * a) for a in range(101)])

    def check(ahist, ehist, aedges, eedges):
      np.testing.assert_array_equal(ahist, ehist)
      np.testing.assert_array_equal(aedges, eedges)

    hist, edges = aupl.df_histogram(df, 'a', 1)
    check(
      hist,   np.array([101]),
      edges,  np.array([0., 100.]))

    hist, edges = aupl.df_histogram(df, 'a', 2)
    check(
      hist,   np.array([50, 51]),
      edges,  np.array([0., 50., 100.]))

    hist, edges = aupl.df_histogram(df, 'b', 4)
    check(
      hist,   np.array([50, 21, 16, 14]),
      edges,  np.array([0, 2500, 5000, 7500, 10000]))


@pytest.mark.slow
def test_histogram_with_examples():
  FIXTURES_BASE_PATH = os.path.join(conf.AU_ROOT, 'au/test/')
  
  def check_fig(fig, fixture_name):
    actual_path = os.path.join(
      testconf.TEST_TEMPDIR_ROOT, 'actual_' + fixture_name)
    print("Saving actual plot to %s" % actual_path)
    aupl.save_bokeh_fig(fig, actual_path, title=fixture_name)
    
    actual_png_path = actual_path.replace('html', 'png')
    print("Saving screenshot of plot to %s" % actual_png_path)
    from bokeh.io import export_png
    export_png(fig, actual_png_path)

    expected_path = os.path.join(FIXTURES_BASE_PATH, fixture_name)
    expected_png_path = expected_path.replace('html', 'png')

    # Compare using PNGs because writing real selenium tests is too much effort
    # for the value at this time.  We tried comparing the raw HTML but bokeh
    # appears to write 
    import imageio
    actual = imageio.imread(actual_png_path)
    expected = imageio.imread(expected_png_path)
    print('Comparing against expected at %s' % expected_png_path)

    np.testing.assert_array_equal(
      actual, expected,
      err_msg=(
        "Page mismatch, actual %s != expected %s, check HTML and PNGs" % (
          actual_path, expected_path)))

  with testutils.LocalSpark.sess() as spark:
    
    # A simple table:
    # +------+------+---+                                                             
    # |mod_11|square|  x|
    # +------+------+---+
    # |     0|     0|  0|
    # |     1|     1|  1|
    # |     2|     4|  2|
    # |     3|     9|  3|
    #    ...
    # +------+------+---+
    from pyspark.sql import Row
    df = spark.createDataFrame([
      Row(x=x, mod_11=int(x % 11), square=x*x)
      for x in range(101)
    ])

    ### Check basic plotting
    pl = aupl.HistogramWithExamplesPlotter()
    fig = pl.run(df, 'x')
    check_fig(fig, 'test_histogram_with_examples_1.html')

    ### Check plotting with custom example plotter
    class PlotterWithMicroFacet(aupl.HistogramWithExamplesPlotter):
      SUB_PIVOT_COL = 'mod_11'
      NUM_BINS = 25

      def display_bucket(self, sub_pivot, bucket_id, irows):
        rows_str = "<br />".join(
            "x: {x} square: {square} mod_11: {mod_11}".format(**row.asDict())
            for row in sorted(irows, key=lambda r: r.x))
        TEMPLATE = """
          <b>Pivot: {spv} Bucket: {bucket_id} </b> <br/>
          {rows}
          <br/> <br/>
        """
        disp = TEMPLATE.format(
                  spv=sub_pivot,
                  bucket_id=bucket_id,
                  rows=rows_str)
        return bucket_id, disp
    
    pl = PlotterWithMicroFacet()
    fig = pl.run(df, 'square')
    check_fig(fig, 'test_histogram_with_examples_2.html')
