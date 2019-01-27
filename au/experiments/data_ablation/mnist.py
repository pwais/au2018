from au.fixtures.tf import mnist

class AblatedDataset(mnist.MNISTDataset):
  
  SPLIT = 'train'
  TARGET_DISTRIBUTION = {}
  UNIFORM_ABLATE = -1
  SEED = 1337

  @classmethod
  def as_imagerow_df(cls, spark):
    df = spark.read.parquet(cls.table_root())
    if cls.SPLIT is not '':
      df = df.filter(df.split == cls.SPLIT)

    if 1.0 >= cls.UNIFORM_ABLATE >= 0:
      df = df.sample(
              withReplacement=False,
              fraction=cls.UNIFORM_ABLATE,
              seed=cls.SEED)
    elif cls.TARGET_DISTRIBUTION:
      df = df.sampleBy(
              "label",
              fractions=cls.TARGET_DISTRIBUTION,
              seed=cls.SEED)
    return df



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