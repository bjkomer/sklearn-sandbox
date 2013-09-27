from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import pylab as pl
import time

class Digits():

  def __init__( self ):
    self.digits = load_digits()
    self.X = self.digits.data
    self.y = self.digits.target
    pl.gray()
    self.fig = pl.figure(1)
    self.ax = self.fig.add_subplot(111)
    self.im = self.ax.matshow(self.digits.images[0])
    self.fig.show()
    self.fig.canvas.draw()

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
      self.clf = svm.LinearSVC()
    elif classifier == "MultinomialNB":
      self.clf = MultinomialNB(alpha=.01)

  def set_train_test_data( self, test_size=10 ):
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

def main():
  digits = load_digits()
  dig = Digits()
  dig.run( test_size=20 )

if __name__ == "__main__":
  main()
