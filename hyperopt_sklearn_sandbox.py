from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import time

from hyperopt import fmin, tpe, hp

class Digits():

  def __init__( self, use_mnist=False ):
    self.use_mnist = use_mnist
    if self.use_mnist:
      #self.digits = fetch_mldata('MNIST original')
      self.mnist_digits_train = fetch_mldata('MNIST original', subset='train')
      self.mnist_digits_test = fetch_mldata('MNIST original', subset='test')
    else:
      self.digits = load_digits()
    self.X = self.digits.data
    self.y = self.digits.target
    self.best_f1_score = 0
    self.best_score = 0
    """
    pl.gray()
    self.fig = pl.figure(1)
    self.ax = self.fig.add_subplot(111)
    self.im = self.ax.matshow(self.digits.images[0])
    self.fig.show()
    self.fig.canvas.draw()
    """

  def display( self, i ):
    self.im.set_data(self.digits.images[i])
    self.fig.canvas.draw()

  def display_range( self, rng ):
    for i in rng:
      self.im.set_data(digits.images[i])
      self.fig.canvas.draw()
      time.sleep(.2)

  def set_classifier( self, classifier ):
    if classifier == "SVC":
      self.clf = svm.SVC()
    elif classifier == "LinearSVC":
      self.clf = svm.LinearSVC( *args, **kwargs )
    elif classifier == "MultinomialNB":
      self.clf = MultinomialNB(alpha=.01)

  def set_train_test_data( self, test_size=100 ):
    if self.use_mnist:
      self.X_train = self.mnist_digits_train.data
      self.y_train = self.mnist_digits_train.target
      self.X_test = self.mnist_digits_test.data
      self.y_test = self.mnist_digits_test.target
    else:
      #np.random.seed(0)
      self.indices = np.random.permutation(len(self.X))
      self.X_train = self.X[ self.indices[:-test_size]]
      self.y_train = self.y[ self.indices[:-test_size]]
      self.X_test = self.X[ self.indices[-test_size:]]
      self.y_test = self.y[ self.indices[-test_size:]]

  def fit( self ):
    self.clf.fit( self.X_train, self.y_train )

  def predict( self ):
    self.prediction = self.clf.predict( self.X_test )

  def compare( self ):
    print( "Test Data" )
    print( self.y_test )
    print( "Prediction" )
    print( self.prediction )

  def display_discrepancy( self ):
    for i in xrange( len( self.y_test ) ):
      if self.y_test[i] != self.prediction[i]:
        self.im.set_data(self.X_test[i].reshape((8,8)))
        self.fig.canvas.draw()
        time.sleep(5)

  def run( self, classifier="LinearSVC", test_size=10 ):
    self.set_train_test_data( test_size=test_size)
    self.set_classifier( classifier=classifier )
    self.fit()
    self.predict()
    self.compare()
    self.display_discrepancy()

  def hyperopt_sklearn_wrapper( self, clf ):
    iterations = 7
    total = 0
    for i in xrange( iterations ):
      self.clf = clf
      self.set_train_test_data()
      self.fit()
      self.predict()
      self.score = metrics.precision_score( self.y_test, self.prediction )
      total += self.score
      if self.score > self.best_score:
        self.best_score = self.score
    return ( iterations - total ) / iterations # Want to minimize this quantity
    

  def hyperopt_wrapper( self, *args, **kwargs ):
    print args
    print kwargs
    iterations = 7
    total = 0
    for i in xrange( iterations ):
      #self.set_classifier( classifier=classifier, *args, **kwargs )
      #self.clf = svm.LinearSVC( *args, **kwargs )
      #self.clf = svm.LinearSVC( **args[0] )
      #self.clf = svm.SVC( **args[0] )
      self.clf = svm.SVC( *args, **kwargs )
      #self.clf = svm.SVC( C=0.3200687726339722, kernel='linear' )
      self.set_train_test_data()
      #self.clf = svm.LinearSVC( C=args[0][0], 
      #                          loss=args[0][1],
      #                          penalty=args[0][2] )
      #self.clf = svm.LinearSVC( C=args[0][0], 
      #                          loss=args[0][1] )
      #self.clf = svm.LinearSVC( C=args[0][0] )
      self.fit()
      self.predict()
      self.f1_score = metrics.f1_score( self.y_test, self.prediction )
      self.score = metrics.precision_score( self.y_test, self.prediction )
      if self.f1_score > self.best_f1_score:
        self.best_f1_score = self.f1_score
        #self.best_C = args[0][0]
      if self.score > self.best_score:
        self.best_score = self.score
        ##self.best_C = args[0][0]
        self.best_C = args[0]['C']
      total += self.score
    # Take an average to prevent overfitting
    return ( iterations - total ) / iterations # Want to minimize this quantity
    #return 1 - self.f1_score # Want to minimize this quantity

  def test_model( self, classifier="LinearSVC", test_size=100, *args, **kwargs ):
    # Use a completely new set to prevent overfitting
    if self.use_mnist:
      X_train = self.mnist_digits_train.data
      y_train = self.mnist_digits_train.target
      X_test = self.mnist_digits_test.data
      y_test = self.mnist_digits_test.target
    else:
      np.random.seed(1)
      indices = np.random.permutation(len(self.X))
      X_train = self.X[ indices[:-test_size]]
      y_train = self.y[ indices[:-test_size]]
      X_test = self.X[ indices[-test_size:]]
      y_test = self.y[ indices[-test_size:]]
    clf = None
    if classifier == "LinearSVC":
      clf = svm.SVC( *args, **kwargs )
      #clf = svm.LinearSVC( **args[0] )
    elif classifier == "SVC":
      #clf = svm.SVC( **args[0] )
      clf = svm.SVC( *args, **kwargs )
    clf.fit( X_train, y_train )
    prediction = clf.predict( X_test )
    f1_score = metrics.f1_score( y_test, prediction )
    score = metrics.precision_score( y_test, prediction )

    #print "F1 Score: %s" % f1_score
    print "Precisions score on test: %s" % score

def main():
  dig = Digits()
  #dig = Digits( use_mnist=True )
  #dig.run( test_size=20 )
  """
  best = fmin( fn=dig.hyperopt_wrapper,
              space={'classifier' : 'LinearSVC',
                     'C' : hp.lognormal('svm_C', 0, 1 ),
                     'loss' : hp.choice('loss', ['l1', 'l2'])
                    },
               algo=tpe.suggest,
               max_evals=50 )
  """
  """
  best = fmin( fn=dig.hyperopt_wrapper,
               space=(
                      hp.lognormal('C', 0, 2 ),
                      hp.choice('loss', ['l1', 'l2']),
                      #hp.choice('penalty', ['l1', 'l2'])
                     ),
               algo=tpe.suggest,
               max_evals=30 )
  """
  """
  best = fmin( fn=dig.hyperopt_wrapper,
               space={
                      'C':hp.lognormal('C', 0, 2 ),
                      'loss':hp.choice('loss', ['l1', 'l2']),
                     },
               algo=tpe.suggest,
               max_evals=30 )
  """
  """
  best = fmin( fn=dig.hyperopt_wrapper,
               space={
                     'C':hp.lognormal('C', 0.0, 2.0 ),
                     'kernel':hp.choice('kernel', ['rbf', 'sigmoid', 'linear']),
                     },
               algo=tpe.suggest,
               max_evals=20 )
  """
  from hpsklearn.components import svc_linear, sklearn_SVC, any_classifier
  #print svc_linear('hai')
  print sklearn_SVC()
  #"""
  best = fmin( fn=dig.hyperopt_sklearn_wrapper,
               #space={'classifier':any_classifier('hai'),'preprocessing':[]},
               space=any_classifier('hai'),
               algo=tpe.suggest,
               max_evals=20 )
  #"""
  """
  best = fmin( fn=dig.hyperopt_wrapper,
               space={
                     'C':hp.lognormal('C', 0, 2 ),
                     'svmkernel':hp.choice('svmkernel', [
                       { 'kernel':'rbf' },
                       { 'kernel':'sigmoid' },
                       { 'kernel':'linear', 
                         'degree':hp.quniform( 'degree', 1, 5, 1 ) } ] ),
                     },
               algo=tpe.suggest,
               max_evals=20 )
  """
  """
  from hpsklearn.components import svc_linear
  best = fmin( fn=dig.hyperopt_wrapper,
               space=svc_linear('name of svc'),
               algo=tpe.suggest,
               max_evals=30 )
  """
  print best
  #print dig.best_f1_score
  print "Precision Score: %s" % dig.best_score
  ##print "C value: %s" % dig.best_C
  #dig.test_model( best, classifier="SVC" )
  # Convert from index to string
  #best['kernel'] = ['rbf', 'sigmoid', 'linear'][best['kernel']]
  dig.test_model( classifier="SVC", **best )
  # 'C': 8.731895397134867e-06 for random seed 0 gets 100% accuracy

if __name__ == "__main__":
  main()


# MNIST with 2 evals:
# {'loss': 0, 'C': 0.3200687726339722}
# Precision Score: 0.900146853147
