from sklearn.datasets import load_digits
from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import time

from hyperopt import fmin, tpe, hp

class Digits():

  def __init__( self ):
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
    np.random.seed(0)
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

  def hyperopt_wrapper( self, *args, **kwargs ):
    #self.set_classifier( classifier=classifier, *args, **kwargs )
    #self.clf = svm.LinearSVC( *args, **kwargs )
    self.set_train_test_data()
    #self.clf = svm.LinearSVC( C=args[0][0], loss=args[0][1] )
    self.clf = svm.LinearSVC( C=args[0][0] )
    self.fit()
    self.predict()
    self.f1_score = metrics.f1_score( self.y_test, self.prediction )
    self.score = metrics.precision_score( self.y_test, self.prediction )
    if self.f1_score > self.best_f1_score:
      self.best_f1_score = self.f1_score
      self.best_C = args[0][0]
    if self.score > self.best_score:
      self.best_score = self.score
      #self.best_C = args[0][0]
    #return 1 - self.score # Want to minimize this quantity
    return 1 - self.f1_score # Want to minimize this quantity

  def test_linearSVC( self, C ):
    clf = svm.LinearSVC( C=C )
    clf.fit( self.X_train, self.y_train )
    prediction = clf.predict( self.X_test )
    f1_score = metrics.f1_score( self.y_test, prediction )
    score = metrics.precision_score( self.y_test, prediction )

    #print f1_score
    print score

def main():
  dig = Digits()
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
  best = fmin( fn=dig.hyperopt_wrapper,
               space=(
                      hp.lognormal('C', 0, 2 ),
               #       hp.choice('loss', ['l1', 'l2'])
                     ),
               algo=tpe.suggest,
               max_evals=75 )

  print best
  print dig.best_f1_score
  #print dig.best_score
  print dig.best_C
  dig.test_linearSVC( C=dig.best_C )
  # 'C': 8.731895397134867e-06 for random seed 0 gets 100% accuracy

if __name__ == "__main__":
  main()
