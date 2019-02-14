import os

from au import util
from au.experiments.data_ablation import util as exp_util
from au.spark import Spark
from au.fixtures.tf import mnist


def gen_ablated_dists(classes, ablations):
  for c in classes:
    for frac in ablations:
      dist = dict((c, 1. / len(classes)) for c in classes)
      dist[c] *= (1. - frac)
      total = sum(dist.itervalues())
      for k in dist.iterkeys():
        dist[k] /= total
      yield dist



class AblatedDataset(mnist.MNISTDataset):
  
  SPLIT = 'train'
  TARGET_DISTRIBUTION = {}
    # Ablate dataset on a per-class basis to these class frequencies
  KEEP_FRAC = -1.0
    # Uniformly ablate the dataset to this fraction
  SEED = 1337

  @classmethod
  def as_imagerow_df(cls, spark):
    df = spark.read.parquet(cls.table_root())
    if cls.SPLIT is not '':
      df = df.filter(df.split == cls.SPLIT)

    if 1.0 >= cls.KEEP_FRAC >= 0:
      df = df.sample(
              withReplacement=False,
              fraction=cls.KEEP_FRAC,
              seed=cls.SEED)
    elif cls.TARGET_DISTRIBUTION:
      df = df.sampleBy(
              "label",
              fractions=cls.TARGET_DISTRIBUTION,
              seed=cls.SEED)
    util.log.info("Ablated to %s examples" % df.count())
    util.log.info("New class frequencies:")
    cls.get_class_freq(spark, df=df).show()

    return df

class ExperimentReport(object):
  def __init__(self, spark, experiment):
    self.spark = spark
    self.experiment = experiment
    self.experiment_df = self.experiment.as_df(self.spark)
  
  def save(self, outdir=None):
    if not outdir:
      row = self.experiment_df.first()
      outdir = row.EXPERIMENT['exp_basedir']
  
    # figs = []
    # figs.extend()
    tabs = self._get_uniform_ablations_figs()
    
    from bokeh import plotting

    plotting.output_file("report.html", title="interactive_legend.py example", mode='inline')
    plotting.save(tabs)

    # from matplotlib.backends.backend_pdf import PdfPages
    # pp = PdfPages(os.path.join(outdir, 'report.pdf'))
    # for fig in figs:
    #   pp.savefig(fig)
    # pp.close()
    # for fig in figs:
    #   plt.close(fig) # Prevent Python from OOMing

  
  
  def _get_uniform_ablations_figs(self, debug=True):
    METRICS = [
      'accuracy',
      'train_accuracy_1',
    ]
    METRICS.extend('precision_%s' % c for c in self.experiment.all_classes)
    METRICS.extend('recall_%s' % c for c in self.experiment.all_classes)

    tags_str = ','.join("'%s'" % t for t in METRICS)

    self.experiment_df.createOrReplaceTempView('experiment')
    df = self.spark.sql("""
      SELECT
        TRAIN_TABLE_KEEP_FRAC keep_frac,
        FIRST(tag) metric_name,
        MAX(simple_value) value,
        params_hash params_hash
      FROM experiment
      WHERE
        tag in ( %s ) AND
        TRAIN_TABLE_KEEP_FRAC >= 0
      GROUP BY TRAIN_TABLE_KEEP_FRAC, tag, params_hash
    """ % tags_str)
    df.createOrReplaceTempView('experiment_uniform_ablations')
    df.cache()

    if debug:
      self.spark.sql("""
        SELECT
          t.metric_name,
          t.keep_frac,
          AVG(100. * t.value) avg,
          STD(100. * t.value) std,
          COUNT(*) support
        FROM experiment_uniform_ablations AS t
        GROUP BY metric_name, keep_frac
        ORDER BY metric_name, keep_frac
      """).show(n=len(METRICS) * len(self.experiment.uniform_ablations))

    from bokeh import plotting
    from bokeh.models.widgets import Panel
    TOOLTIPS = [
        ("(x,y)", "($x{0.000000}, $y)"),
    ]
    panels = []

    def plot_scatter(fig, metric_name, color='blue', alpha=0.2, label=None):
      assert metric_name in METRICS # Programming error?
      util.log.info("Plotting scatter %s ..." % metric_name)
      from bokeh.models import ColumnDataSource
      df = self.spark.sql("""
                  SELECT keep_frac, value
                  FROM experiment_uniform_ablations
                  WHERE metric_name = '%s'
                """ % metric_name).toPandas()
      source = ColumnDataSource(df)
      fig.circle(
        x='keep_frac', y='value', source=source,
        size=10,
        color=color, alpha=alpha, legend=label)
      
      # Add whiskers
      errors = []
      for keep_frac in df.keep_frac.unique():
        values = df[df['keep_frac'] == keep_frac]['value']
        mu = values.mean()
        std = values.std()
        errors.append({
          'base': keep_frac,
          'lower': mu - std,
          'upper': mu + std,
        })

      import pandas as pd
      from bokeh.models import Whisker
      w = Whisker(
          source=ColumnDataSource(pd.DataFrame(errors)),
          base='base', lower='lower', upper='upper',
          line_color='dark' + color)
      fig.add_layout(w)

    
    ### All-class metrics
    ## Test Accuracy
    fig = plotting.figure(
              tooltips=TOOLTIPS, plot_width=1000,
              x_axis_type='log',
              title='Test Accuracy vs [Log-scale] Ablated Training Set Size')
    plot_scatter(fig, 'accuracy')
    fig.xaxis.axis_label = 'Fraction of Training Set'
    fig.yaxis.axis_label = 'Test Accuracy'
    panels.append(Panel(child=fig, title="Test Accuracy"))

    
    ## Train Accuracy
    fig = plotting.figure(
              tooltips=TOOLTIPS, plot_width=1000,
              title='Train Accuracy vs Ablated Training Set Size')
    plot_scatter(fig, 'train_accuracy_1')
    fig.xaxis.axis_label = 'Fraction of Training Set'
    fig.yaxis.axis_label = 'Train Accuracy'
    panels.append(Panel(child=fig, title="Train Accuracy"))
    

    ### Per-class metrics
    for c in self.experiment.all_classes:
      title = 'Precision / Recall vs Ablated Training Set Size (Class: %s)' % c
      fig = plotting.figure(title=title, x_axis_type='log',
                tooltips=TOOLTIPS, plot_width=1000)
      plot_scatter(fig, 'precision_%s' % c, color='blue', label='Precision')
      plot_scatter(fig, 'recall_%s' % c, color='red', label='Recall')

      fig.xaxis.axis_label = 'Fraction of Training Set'
      fig.yaxis.axis_label = 'Metric'

      fig.legend.location = 'bottom_right'
      fig.legend.click_policy = 'hide'

      panels.append(Panel(child=fig, title="P/R Class %s" % c))

    from bokeh.models.widgets import Tabs
    return Tabs(tabs=panels)


  
  def _get_uniform_ablations_plot(self, spark, experiment_df):
    df = _get_uniform_ablations_df
  


class Experiment(object):
  DEFAULTS = {
    'exp_basedir': exp_util.experiment_basedir('mnist'),
    'run_name': 'default.' + util.fname_timestamp(),
    #'default.2019-02-03-07_25_48.GIBOB', #

    'params_base':
      mnist.MNIST.Params(
        TRAIN_EPOCHS=100,
        TRAIN_WORKER_CLS=util.WholeMachineWorker,
      ),
    
    'trials_per_treatment': 10,

    'uniform_ablations': #tuple(),
    (
        0.9999,
        0.9995,
        0.999,
        0.995,
        0.99,
        0.95,
        0.9,
        0.5,
        0.0,
      ),
    
    'single_class_ablations': tuple(),
    # (
    #   0.9999,
    #   0.999,
    #   0.99,
    #   0.9,
    #   0.5,
    # ),
    'all_classes': range(10),
  }

  @property
  def run_dir(self):
    return os.path.join(self.exp_basedir, self.run_name)

  def __init__(self, **conf):
    for k, v in self.DEFAULTS.iteritems():
      setattr(self, k, v)
    for k, v in conf.iteritems():
      setattr(self, k, v)

  def run(self, spark=None):
    ps = list(self._iter_model_params())

    with Spark.sess(spark) as spark:
      
      
      class ModelFactory(object):
        def __init__(self, p, m):
          self.params = p
          self.meta = m
        
        def __call__(self):
          model = mnist.MNIST.load_or_train(params=self.params)

          meta_path = os.path.join(self.params.MODEL_BASEDIR, 'au_meta.json')
          if not os.path.exists(meta_path):
            with open(meta_path, 'wc') as f:
              import json
              json.dump(self.meta, f)
            util.log.info("Saved meta to %s" % meta_path)
          
          return model

      callables = (
        ModelFactory(p, self._get_params_meta(p))
        for p in ps)
      res = Spark.run_callables(spark, callables)
    


  def as_df(self, spark, include_tf_summaries=True):
    paths = util.all_files_recursive(self.run_dir, pattern='**/au_meta.json')
    df = spark.read.json(paths)
    if include_tf_summaries:
      from pyspark.sql import Row
      
      def add_params_hash(meta_row):
        d = meta_row.asDict()
        d['params_hash'] = hash(str(meta_row))
        return Row(**d)
      meta_df = spark.createDataFrame(
                          df.rdd.map(add_params_hash),
                          samplingRatio=1)

      def meta_to_rows(meta_row):
        model_dir = meta_row.MODEL_BASEDIR
        if os.path.exists(model_dir):
          for r in util.TFSummaryReader(glob_events_from_dir=model_dir):
            yield r.as_row(extra={'params_hash': meta_row.params_hash})
      tf_summary_df = spark.createDataFrame(meta_df.rdd.flatMap(meta_to_rows))

      df = meta_df.join(tf_summary_df, on='params_hash', how='inner')
        # This join is super fast because `meta_df` is quite small.  Moreover,
        # Spark is smart enough not to compute / use extra memory for duplicate
        # values that would result from instantiating copies of `meta_df`
        # rows in the join.
    return df

  


  # def create_tf_summary_df(self, spark):
  #   df = exp_util.params_to_tf_summary_df(spark, self.iter_model_params())
  #   # df = Spark.union_dfs(*(
  #   #   exp_util.params_to_tf_summary_df(spark, p)
  #   #   for p in self.iter_model_params()))
  #   return df

  def _get_params_meta(self, params):
    d = util.as_row_of_constants(params)
    exp_dict = dict((k, getattr(self, k)) for k in self.DEFAULTS.iterkeys())
    exp_dict['params_base'] = util.as_row_of_constants(['params_base'])
    d['EXPERIMENT'] = exp_dict
    return d


  def _iter_model_params(self):
    import copy

    ## Uniform Ablations
    for ablate_frac in self.uniform_ablations:
      keep_frac = 1.0 - ablate_frac
      params = copy.deepcopy(self.params_base)

      params.TRAIN_WORKER_CLS = util.SingleGPUWorker

      for i in range(self.trials_per_treatment):
        params = copy.deepcopy(params)

        class ExpTable(AblatedDataset):
          KEEP_FRAC = keep_frac
          SEED = AblatedDataset.SEED + i
        params.TRIAL = i
        params.TRAIN_TABLE = ExpTable
        params.MODEL_NAME = (
          'ablated_mnist_keep_%s' % keep_frac + '_trial_' + str(i))
        params.MODEL_BASEDIR = os.path.join(
                                self.exp_basedir,
                                self.run_name,
                                params.MODEL_NAME)
        yield params
    
    ## Per-class Ablations
    for dist in gen_ablated_dists(self.all_classes, self.single_class_ablations):
      params = copy.deepcopy(self.params_base)

      params.TRAIN_WORKER_CLS = util.WholeMachineWorker

      ablated_frac, ablated_class = min((frac, c) for c, frac in dist.iteritems())

      for i in range(self.trials_per_treatment):
        params = copy.deepcopy(params)

        class ExpTable(AblatedDataset):
          TARGET_DISTRIBUTION = dist
          SEED = AblatedDataset.SEED + i
        params.TRIAL = i
        params.TRAIN_TABLE = ExpTable
        params.MODEL_NAME = (
          ('ablated_mnist_class_%s_keep_%s' % (ablated_class, ablated_frac)) + '_trial_' + str(i))
        params.MODEL_BASEDIR = os.path.join(
                                self.exp_basedir,
                                self.run_name,
                                params.MODEL_NAME)
        yield params
      



"""
need to extract generator -> tf.Dataset thingy to make ImageTables have a
as_tf_dataset() method mebbe ?

then just generate ablations and train.

for test, we just use same test set and can do test ablations after the fact

finally let's do activation tables of each model and see if we can 
generate embeddings with each model and then plot test data in those embeddings.

so, like, maybe there's a function(model, embedding, test_data_slice_1) that predicts
error on test_data_slice_2.  

build the above with mscoco / segmentation in mind, as well as bdd100k segmentation


"""

if __name__ == '__main__':
  mnist.MNISTDataset.setup()
  Experiment().run()

