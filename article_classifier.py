from sklearn.datasets import fetch_20newsgroups

#categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space' ]
categories = ['alt.atheism', 'comp.graphics', 'sci.space' ]

newsgroups_train = fetch_20newsgroups( subset='train', 
                                       categories=categories,)
#                                       remove=( 'headers', 'footers', 'quotes' ) )
newsgroups_test = fetch_20newsgroups( subset='test', 
                                      categories=categories,)
#                                      remove=( 'headers', 'footers', 'quotes' ) )

from pprint import pprint

pprint( list( newsgroups_train.target_names ) )
pprint( newsgroups_test.keys() )

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform( newsgroups_train.data )

from sklearn import svm, metrics
from sklearn.linear_model import SGDClassifier

vectors_test = vectorizer.transform( newsgroups_test.data )

#####

#clf = svm.LinearSVC( multi_class='crammer_singer' )
#clf = svm.LinearSVC( loss='l1', penalty='l2', dual=True )
clf = svm.LinearSVC( )
#clf = svm.SVC( kernel='rbf' )
#clf = svm.SVC( kernel='linear' )
#clf = svm.SVC( kernel='sigmoid' )
#clf = svm.SVC( kernel='poly', degree=3 )
#clf = SGDClassifier(loss="hinge", penalty="l2")

#####

clf.fit( vectors, newsgroups_train.target )
print( vectors.shape )
pred = clf.predict( vectors_test )
print( metrics.f1_score( newsgroups_test.target, pred ) )
"""
import numpy as np
def show_top( classifier, vectorizer, categories, n=10 ):
  feature_names = np.asarray( vectorizer.get_feature_names() )
  for i, category in enumerate( categories ):
    top = np.argsort( classifier.coef_[i])[-n:]
    print( "%s: %s" % (category, " ".join( feature_names[ top ] ) ) )

show_top( clf, vectorizer, newsgroups_train.target_names, n=10 )
"""
