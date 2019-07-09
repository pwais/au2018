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
      yield dist

class ExperimentWorker(util.SingleGPUWorker):
  GPU_POOL = util.GPUPool() # Use a pool for the current experiment run

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
    run_name = self.experiment.run_name
    outdir = self.experiment.run_dir
  
    def save_plot(tabs, fname, title):
      from bokeh import plotting
      if tabs is None:
        return
      
      dest = os.path.join(outdir, fname)
      plotting.output_file(dest, title=title + ' - ' + run_name, mode='inline')
      plotting.save(tabs)
      util.log.info("Wrote to %s" % dest)

    tabs = self._get_uniform_ablations_figs()
    save_plot(tabs, 'report_uniform.html', 'Uniform Ablations')
    
    tabs = self._get_single_class_ablations_plots()
    save_plot(tabs, 'report_single_class.html', 'Single-Class Ablations')

    util.log.info("Saved reports to %s" % outdir)
  
  def _get_uniform_ablations_figs(self, debug=True):
    # Skip entirely if no data or experiments
    if set(['tag', 'TRAIN_TABLE_KEEP_FRAC']) - set(self.experiment_df.columns):
      util.log.info("Skipping uniform ablations")
      return None

    util.log.info("Plotting uniform ablation figs ...")
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
    from bokeh.models import ColumnDataSource
    from bokeh.models.widgets import Panel
    TOOLTIPS = [
        ("(x,y)", "($x{0.000000}, $y)"),
    ]
    panels = []

    def plot_scatter(fig, metric_name, color='blue', alpha=0.2, label=None):
      assert metric_name in METRICS # Programming error?
      util.log.info("Plotting scatter %s ..." % metric_name)
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

      return df

    
    ### All-class metrics
    ## Test Accuracy
    fig = plotting.figure(
              tooltips=TOOLTIPS, plot_width=1000,
              x_axis_type='log',
              title='Test Accuracy vs [Log-scale] Ablated Training Set Size')
    acc_df = plot_scatter(fig, 'accuracy')
    fig.xaxis.axis_label = 'Fraction of Training Set (m)'
    fig.yaxis.axis_label = 'Test Accuracy'
    panels.append(Panel(child=fig, title="Test Accuracy"))

    import numpy as np
    import scipy.optimize
    def f(m, c):
      # PAC loose bound when Acc_Train is 1:
      # Acc_Test > 1 - sqrt( (logH + log(1/delta))  / 2m)
      return 1. - np.sqrt(c / m)
    
    xdata = tuple(acc_df.keep_frac)
    ydata = tuple(acc_df.value)
    popt, pcov = scipy.optimize.curve_fit(f, xdata, ydata)
    
    c = popt[0]
    N = 1e4 + 10
    xs = np.linspace(0, 1, N)
    ys = [f(x, c) for x in xs]
    ys = [(y if y >= 0 else 0) for y in ys]
    fig.line(
      x=xs, y=ys, line_width=3, alpha=0.5,
      legend='Acc(m) = 1 - sqrt(c / m), c = %s' % c)
    
    fig.legend.location = 'bottom_right'
    fig.legend.click_policy = 'hide'

    
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

    ### Training Runtime
    util.log.info("Plotting runtime ...")
    walltime_sdf = self.spark.sql("""
      SELECT
        TRAIN_TABLE_KEEP_FRAC keep_frac,
        params_hash params_hash,
        MAX(wall_time) - MIN(wall_time) train_time
      FROM experiment
      WHERE
        TRAIN_TABLE_KEEP_FRAC >= 0
      GROUP BY TRAIN_TABLE_KEEP_FRAC, params_hash
    """)
    walltime_df = walltime_sdf.toPandas()
    fig = plotting.figure(
              tooltips=TOOLTIPS, plot_width=1000, x_axis_type='log',
              title='Train Time vs Ablated Training Set Size')
    fig.circle(
      x='keep_frac', y='train_time', source=ColumnDataSource(walltime_df),
      size=10)
    fig.xaxis.axis_label = 'Fraction of Training Set'
    fig.yaxis.axis_label = 'Train Time (sec)'
    panels.append(Panel(child=fig, title="Train Walltime"))

    from bokeh.models.widgets import Tabs
    return Tabs(tabs=panels)


  
  def _get_single_class_ablations_plots(self, debug=True):
    # Skip entirely if no data or experiments
    if set(['tag', 'TRAIN_TABLE_TARGET_DISTRIBUTION']) - \
            set(self.experiment_df.columns):
      util.log.info("Skipping single-class ablations")
      return None

    util.log.info("Plotting single-class ablation figs ...")
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
        TRAIN_TABLE_TARGET_DISTRIBUTION class_dist,
        FIRST(tag) metric_name,
        MAX(simple_value) value,
        params_hash params_hash
      FROM experiment
      WHERE
        tag in ( %s ) AND
        TRAIN_TABLE_KEEP_FRAC < 0
      GROUP BY TRAIN_TABLE_TARGET_DISTRIBUTION, tag, params_hash
    """ % tags_str)
    
    from pyspark.sql.functions import udf
    from pyspark.sql import types
    def get_ablated_class(class_dist):
      # pyspark gives us a Row() instead of a dict :P
      from pyspark.sql import Row
      if isinstance(class_dist, Row):
        class_dist = class_dist.asDict()
      min_key, min_value = min(iter(class_dist.items()), key=lambda k_v: k_v[-1])
      return Row(class_id=str(min_key), keep_frac=min_value)
    
    get_ablated_class_udf = udf(
      get_ablated_class,
      types.StructType([
        types.StructField("class_id", types.StringType(), False),
        types.StructField("keep_frac", types.FloatType(), False)
      ]))

    df = df.withColumn('ablated_class', get_ablated_class_udf(df.class_dist))
    df.createOrReplaceTempView('experiment_single_class_ablations')
    df.cache()
    

    
    from bokeh import plotting
    from bokeh.models import ColumnDataSource
    from bokeh.models.widgets import Panel
    TOOLTIPS = [
        ("(x,y)", "($x{0.000000}, $y)"),
    ]
    panels = []

    ALPHA = 0.2

    ### Target Class vs Others
    ## Test Accuracy
    fig = plotting.figure(
      tooltips=TOOLTIPS, plot_width=1000,
      x_axis_type='log',
      title='Test Accuracy vs [Log-scale] Class-Ablated Training Set Size')
    for class_id in self.experiment.all_classes:
      target_class = class_id
      other_classes = set(self.experiment.all_classes) - set([target_class])

      util.log.info("Plotting %s Test Accuracy ..." % target_class)
      df = self.spark.sql("""
                    SELECT ablated_class.keep_frac, value
                    FROM experiment_single_class_ablations
                    WHERE
                      metric_name = 'accuracy' AND
                      ablated_class.class_id = '%s'
                  """ % target_class).toPandas()
      source = ColumnDataSource(df)
      fig.circle(
          x='keep_frac', y='value', source=source,
          size=10,
          color=util.hash_to_rbg(target_class),
          alpha=ALPHA,
          legend='%s ablated' % target_class)
    fig.legend.location = 'bottom_right'
    fig.legend.click_policy = 'hide'
    fig.xaxis.axis_label = 'Fraction of Training Set'
    fig.yaxis.axis_label = 'Test Accuracy'
    panels.append(Panel(child=fig, title="Test Accuracy"))

    ## Relative Precision and Recall
    for class_id in sorted(self.experiment.all_classes):
      target_class = class_id
      other_classes = set(self.experiment.all_classes) - set([target_class])

      def plot_metric(metric_prefix):
        util.log.info("Plotting %s %s ..." % (target_class, metric_prefix))
        # Target class
        metric_name = '%s_%s' % (metric_prefix, target_class)
        df = self.spark.sql("""
                    SELECT ablated_class.keep_frac, value
                    FROM experiment_single_class_ablations
                    WHERE
                      metric_name = '%s' AND
                      ablated_class.class_id = '%s'
                  """ % (metric_name, target_class)).toPandas()
        source = ColumnDataSource(df)
        fig.circle(
            x='keep_frac', y='value', source=source,
            size=10,
            color=util.hash_to_rbg(target_class),
            alpha=ALPHA,
            legend='%s ablated' % target_class)
        
        # Other classes
        metric_names = ('%s_%s' % (metric_prefix, c) for c in other_classes)
        metric_names_clause = ','.join("'%s'" % n for n in metric_names)
        df = self.spark.sql("""
                    SELECT
                      ablated_class.keep_frac,
                      AVG(value) mean_value,
                      AVG(value) - STD(value) std_lower,
                      AVG(value) + STD(value) std_upper
                    FROM experiment_single_class_ablations
                    WHERE
                      metric_name in ( %s ) AND
                      ablated_class.class_id = '%s'
                    GROUP BY ablated_class.keep_frac
                  """ % (metric_names_clause, target_class)).toPandas()
        source = ColumnDataSource(df)
        fig.circle(
            x='keep_frac', y='mean_value', source=source,
            size=10,
            color='black',
            alpha=ALPHA,
            legend='Non-%s classes (Mean)' % target_class)
        
        from bokeh.models import Whisker
        w = Whisker(
              source=source,
              base='mean_value', lower='std_lower', upper='std_upper',
              line_color='black')
        fig.add_layout(w)

      # Precision
      fig = plotting.figure(
        tooltips=TOOLTIPS, plot_width=1000,
        x_axis_type='log',
        title='Precision vs [Log-scale] Class-Ablated Training Set Size')
      plot_metric('precision')
      fig.xaxis.axis_label = (
        'Fraction of %s Retained in Training Set' % target_class)
      fig.yaxis.axis_label = 'Precision'
      panels.append(Panel(child=fig, title="Precision %s" % target_class))

      # Recall
      fig = plotting.figure(
        tooltips=TOOLTIPS, plot_width=1000,
        x_axis_type='log',
        title='Recall vs [Log-scale] Class-Ablated Training Set Size')
      plot_metric('recall')
      fig.xaxis.axis_label = (
        'Fraction of %s Retained in Training Set' % target_class)
      fig.yaxis.axis_label = 'Recall'
      panels.append(Panel(child=fig, title="Recall %s" % target_class))


    ### Overall Metrics
    ## Train Accuracy
    fig = plotting.figure(
              tooltips=TOOLTIPS, plot_width=1000,
              title='Train Accuracy vs Ablated Training Set Size')
    for class_id in self.experiment.all_classes:
      target_class = class_id

      util.log.info("Plotting %s Train Accuracy ..." % target_class)
      df = self.spark.sql("""
                    SELECT ablated_class.keep_frac, value
                    FROM experiment_single_class_ablations
                    WHERE
                      metric_name = 'train_accuracy_1' AND
                      ablated_class.class_id = '%s'
                  """ % target_class).toPandas()
      source = ColumnDataSource(df)
      fig.circle(
          x='keep_frac', y='value', source=source,
          size=10,
          color=util.hash_to_rbg(target_class),
          alpha=ALPHA,
          legend='%s ablated' % target_class)
    fig.legend.location = 'bottom_right'
    fig.legend.click_policy = 'hide'
    fig.xaxis.axis_label = 'Fraction of Training Set'
    fig.yaxis.axis_label = 'Train Accuracy'
    panels.append(Panel(child=fig, title="Train Accuracy"))
    
    ### Training Runtime
    util.log.info("Plotting runtime ...")
    fig = plotting.figure(
              tooltips=TOOLTIPS, plot_width=1000,
              title='Train Times')
    walltime_sdf = self.spark.sql("""
      SELECT
        params_hash params_hash,
        MAX(wall_time) - MIN(wall_time) train_time
      FROM experiment
      WHERE
        TRAIN_TABLE_KEEP_FRAC < 0
      GROUP BY params_hash
    """)
    walltimes = [row.train_time for row in walltime_sdf.collect()]
    import numpy as np
    hist, edges = np.histogram(walltimes, density=True, bins=50)
    fig.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], alpha=0.4)
    fig.xaxis.axis_label = 'Train Time (sec)'
    fig.yaxis.axis_label = 'Density'
    panels.append(Panel(child=fig, title="Train Walltime"))

    from bokeh.models.widgets import Tabs
    return Tabs(tabs=panels)




class Experiment(object):
  DEFAULTS = {
    'exp_basedir': exp_util.experiment_basedir('mnist'),
    'run_name': 'default.' + util.fname_timestamp(),

    'params_base':
      mnist.MNIST.Params(
        TRAIN_EPOCHS=2,
        TRAIN_WORKER_CLS=ExperimentWorker,
      ),
    
    'trials_per_treatment': 10,

    'uniform_ablations': (
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
    
    'single_class_ablations': (
      0.9999,
      0.999,
      0.99,
      0.9,
      0.5,
    ),

    'all_classes': list(range(10)),
  }

  @property
  def run_dir(self):
    return os.path.join(self.exp_basedir, self.run_name)

  def __init__(self, **conf):
    for k, v in self.DEFAULTS.items():
      setattr(self, k, conf.get(k) or v)

  def run(self, spark=None):
    self._train_models(spark=spark)

    # TODO run reports
    # self._build_activations(spark=spark)

    self._save_reports(spark=spark)

  def _iter_activation_tables(self):
    from au.fixtures import nnmodel
    for params in self._iter_model_params():
      class TreatmentTestActivations(nnmodel.ActivationsTable):
        SPLIT = 'test'
        TABLE_NAME = params.MODEL_NAME + '_test_activations'
        NNMODEL_CLS = mnist.MNIST
        MODEL_PARAMS = params
        IMAGE_TABLE_CLS = params.TEST_TABLE
      yield TreatmentTestActivations

      class TreatmentFullTrainActivations(nnmodel.ActivationsTable):
        SPLIT = 'train'
        TABLE_NAME = params.MODEL_NAME + '_train_activations'
        NNMODEL_CLS = mnist.MNIST
        MODEL_PARAMS = params
        IMAGE_TABLE_CLS = mnist.MNIST.Params().TRAIN_TABLE
      yield TreatmentFullTrainActivations

  def _build_activations(self, spark=None):
    tables = list(self._iter_activation_tables())
    util.log.info("Building activation tables ...")
    with Spark.sess(spark) as spark:
      for i, t in enumerate(tables):
        t.setup(spark=spark)
        t.save_tf_embedding_projector(spark=spark)
        util.log.info(
          "... completed %s / %s (%s) ..." % (
            i + 1, len(tables), t.TABLE_NAME))


  def _train_models(self, spark=None):
    util.log.info("Training models ...")
    import time
    start = time.time()

    ps = list(self._iter_model_params())

    with Spark.sess(spark) as spark:
  
      class TreatmentFactory(object):
        def __init__(self, p, m):
          self.params = p
          self.meta = m
        
        def __call__(self):
          model = mnist.MNIST.load_or_train(params=self.params)

          # For now, dump treatment metadata to disk
          meta_path = os.path.join(self.params.MODEL_BASEDIR, 'au_meta.json')
          if not os.path.exists(meta_path):
            with open(meta_path, 'wc') as f:
              import json
              json.dump(self.meta, f)
            util.log.info("Saved meta to %s" % meta_path)
          
          return model

      callables = (
        TreatmentFactory(p, self._get_params_meta(p))
        for p in ps)
      res = Spark.run_callables(spark, callables)
    
    util.log.info(
      "Model training complete in %s mins. Saved to %s" % (
      (time.time() - start) / 60., self.run_dir))

  def _save_reports(self, spark=None):
    util.log.info("Generating reports ...")
    with Spark.sess(spark) as spark:
      report = ExperimentReport(spark=spark, experiment=self)
      report.save()

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
      tf_summary_rdd = meta_df.rdd.flatMap(meta_to_rows)
      if tf_summary_rdd.isEmpty():
        df = meta_df
      else:
        tf_summary_df = spark.createDataFrame(tf_summary_rdd)
        df = meta_df.join(tf_summary_df, on='params_hash', how='inner')
          # This join is super fast because `meta_df` is quite small. 
          # Moreover, Spark is smart enough not to compute / use extra memory
          # for duplicate values that would result from instantiating copies
          # of `meta_df` rows in the join.
    return df

  


  # def create_tf_summary_df(self, spark):
  #   df = exp_util.params_to_tf_summary_df(spark, self.iter_model_params())
  #   # df = Spark.union_dfs(*(
  #   #   exp_util.params_to_tf_summary_df(spark, p)
  #   #   for p in self.iter_model_params()))
  #   return df

  def _get_params_meta(self, params):
    d = util.as_row_of_constants(params)
    exp_dict = dict((k, getattr(self, k)) for k in self.DEFAULTS.keys())
    exp_dict['params_base'] = util.as_row_of_constants(['params_base'])
    d['EXPERIMENT'] = exp_dict
    return d


  def _iter_model_params(self):
    import copy

    ## Uniform Ablations
    for ablate_frac in self.uniform_ablations:
      keep_frac = 1.0 - ablate_frac
      params = copy.deepcopy(self.params_base)

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

      ablated_frac, ablated_class = min((frac, c) for c, frac in dist.items())

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
  import sys
  mnist.MNISTDataset.setup()
  
  print("Example usage: python mnist.py run_name=my_run")

  kv_args = dict(a.split('=') for a in sys.argv if '=' in a)
  e = Experiment(**kv_args)
  e.run()

