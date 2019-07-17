from au import plotting as aupl
from au.test import testconf
from au.test import testutils

import numpy as np

def test_hash_to_rgb():
  assert aupl.hash_to_rbg('moof') == (204, 40, 144)
  assert aupl.hash_to_rbg(5) == (204, 155, 40)
  assert aupl.hash_to_rbg('moof1') != aupl.hash_to_rbg('moof')


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


def test_histogram_with_examples_basic():
  with testutils.LocalSpark.sess() as spark:
    from pyspark.sql import Row
    df = spark.createDataFrame([
      Row(x=x, mod_11=int(x % 11), square=x*x)
      for x in range(101)
    ])

    pl = aupl.HistogramWithExamplesPlotter()
    fig = pl.run(df, 'x')
    aupl.save_bokeh_fig(fig, '/opt/au/tasttast.html')

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
    aupl.save_bokeh_fig(fig, '/opt/au/tasttast2.html')

