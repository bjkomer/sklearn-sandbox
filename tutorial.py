from __future__ import print_function
import numpy as np
from sklearn import datasets

def supervised_learning():
  iris = datasets.load_iris()
  iris_X = iris.data
  iris_y = iris.target
  np.unique(iris_y)


  # Split iris data in train and test data
  # A random permutation, to split the data randomly
  np.random.seed(0)
  indices = np.random.permutation(len(iris_X))
  iris_X_train = iris_X[ indices[:-10] ]
  iris_y_train = iris_X[ indices[:-10] ]
  iris_X_test = iris_X[ indices[-10:] ]
  iris_y_test = iris_y[ indices[-10:] ]
  # Create and fit a nearest-neighbor classifier
  from sklearn.neighbors import KNeighborsClassifier
  knn = KNeighborsClassifier()
  print( knn.fit( iris_X_train, iris_y_train ) )
  print( knn.predict( iris_X_test ) )
  print( iris_y_test )

def regression():
  diabetes = datasets.load_diabetes()
  diabetes_X_train = diabetes.data[:-20]
  diabetes_X_test  = diabetes.data[-20:]
  diabetes_y_train = diabetes.target[:-20]
  diabetes_y_test  = diabetes.target[-20:]
  
  from sklearn import linear_model
  regr = linear_model.LinearRegression()
  print( regr.fit( diabetes_X_train, diabetes_y_train ) )
  print( regr.coef_ )
  # The mean square error
  print( np.mean( ( regr.predict( diabetes_X_test ) - diabetes_y_test ) **2 ) )
  # Explained variance score: 1 is perfect prediction
  # and 0 means that there is no linear relationship
  # between X and Y.
  print( regr.score( diabetes_X_test, diabetes_y_test ) )

  # Shrinkage

  X = np.c_[ 0.5, 1 ].T
  y = [ 0.5, 1 ]

  test = np.c_[ 0, 2 ].T
  regr = linear_model.LinearRegression()

  import pylab as pl
  pl.figure()

  np.random.seed(0)
  for _ in range(6):
    this_X = .1 * np.random.normal( size=(2,1) ) + X
    regr.fit( this_X, y )
    pl.plot( test, regr.predict( test ) )
    pl.scatter( this_X, y, s=3 )

  regr = linear_model.Ridge( alpha=0.1 )
  pl.figure()
  np.random.seed(0)
  for _ in range(6):
    this_X = .1 * np.random.normal( size=(2,1) ) + X
    regr.fit( this_X, y )
    pl.plot( test, regr.predict( test ) )
    pl.scatter( this_X, y, s=3 )

  alphas = np.logspace( -4, -1, 6 )
  #from __future__ import print_function
  print([regr.set_params(alpha=alpha
    ).fit(diabetes_X_train, diabetes_y_train,
    ).score(diabetes_X_test, diabetes_y_test) for alpha in alphas])

  # Sparsity

  regr = linear_model.Lasso()
  scores = [ regr.set_params( alpha=alpha
    ).fit(diabetes_X_train, diabetes_y_train
    ).score(diabetes_X_test, diabetes_y_test)
    for alpha in alphas ]
  best_alpha = alphas[scores.index(max(scores))]
  regr.alpha = best_alpha
  print( regr.fit(diabetes_X_train, diabetes_y_train) )
  print( regr.coef_ )

  # Classification

  logistic = linear_model.LogisticRegression( C = 1e5 )
  print( logistic.fit( iris_X_train, iris_y_train ) )



def main():
  #supervised_learning()
  regression()

if __name__ == "__main__":
  main()
