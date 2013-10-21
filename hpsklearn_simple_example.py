from hpsklearn.estimator import hyperopt_estimator
from hpsklearn.components import any_classifier
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_mldata
from sklearn import svm
from sklearn import metrics
import numpy as np

from hyperopt import fmin, tpe, hp

def score( y1, y2 ):
  length = len( y1 )
  correct = 0.0
  for i in xrange(length):
    if y1[i] == y2[i]:
      correct += 1.0
  return correct / length

def sklearn_digits( classifier=None ):
  #estim = hyperopt_estimator( classifier=any_classifier('hai'), algo=tpe.suggest )
  if classifier is None:
    classifier = any_classifier('any')
  estim = hyperopt_estimator( classifier=classifier )

  digits = load_digits()
  X = digits.data
  y = digits.target

  test_size = 50
  np.random.seed(0)
  indices = np.random.permutation(len(X))
  X_train = X[ indices[:-test_size]]
  y_train = y[ indices[:-test_size]]
  X_test = X[ indices[-test_size:]]
  y_test = y[ indices[-test_size:]]

  estim.fit( X_train, y_train )

  pred = estim.predict( X_test )
  print( pred )
  print ( y_test )

  print( score( pred, y_test ) ) 
  
  print( estim.best_model() )

def mnist_digits():
  estim = hyperopt_estimator( classifier=any_classifier('hai') )

  digits = fetch_mldata('MNIST original')
  X = digits.data
  y = digits.target

  test_size = int( 0.2 * len( y ) )
  np.random.seed(0)
  indices = np.random.permutation(len(X))
  X_train = X[ indices[:-test_size]]
  y_train = y[ indices[:-test_size]]
  X_test = X[ indices[-test_size:]]
  y_test = y[ indices[-test_size:]]

  estim.fit( X_train, y_train )

  pred = estim.predict( X_test )
  print( pred )
  print ( y_test )

  print( score( pred, y_test ) ) 

  print( estim.best_model() )

from hpsklearn.components import svc
from hpsklearn.components import liblinear_svc
from hpsklearn.components import knn
from hpsklearn.components import random_forest
from hpsklearn.components import extra_trees

#sklearn_digits()
sklearn_digits( classifier=extra_trees('hai') )
#mnist_digits()
