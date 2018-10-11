

class ActivationObserver(object):
  
  def record(self, activations):
    pass
  
  def record_multiple(self, act_seq):
    pass

class ActivationDistribution(object):
  
  def tally(self, activation):
    pass
  
  @staticmethod
  def load_from_something(foo):
    pass

  def compute_prob_of(self, act):
    pass

"""

1) spark-able task to run inference and hit an observer
2) spark job to run mnist inference and record raw activations

3) 

experiments:
A)
 * train mnist network
 * create random ferns model on the upper layer
 * how does random ferns do on test set?  how does it do on in- vs out-of-distro?

B)
 * use LSH to create hash of LSH -> container(activation, class, ...)
 * does running test on model help retrieve correct class? or really similar
      examples?

C)
 * build parzan windows to represent whole joint / all training activations.
     how much do random ferns and parzan probs differ when computing probs
     on test set?



"""