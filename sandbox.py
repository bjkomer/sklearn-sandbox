from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space' ]
"""
newsgroups_train = fetch_20newsgroups( subset='train', 
                                       categories=categories,
                                       remove=( 'headers', 'footers', 'quotes' ) )
newsgroups_test = fetch_20newsgroups( subset='test', 
                                      categories=categories,
                                      remove=( 'headers', 'footers', 'quotes' ) )
"""

newsgroups_train = fetch_20newsgroups( subset='train', 
                                       categories=categories )
newsgroups_test = fetch_20newsgroups( subset='test', 
                                      categories=categories )

from pprint import pprint

pprint( list( newsgroups_train.target_names ) )

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform( newsgroups_train.data )


from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

vectors_test = vectorizer.transform( newsgroups_test.data )
clf = MultinomialNB( alpha=0.01 )
clf.fit( vectors, newsgroups_train.target )
pred = clf.predict( vectors_test )
print( vectors.shape )
print( vectors.nnz / float( vectors.shape[0] ) )
print( metrics.f1_score( newsgroups_test.target, pred ) )

import numpy as np
def show_top( classifier, vectorizer, categories, n=10 ):
  feature_names = np.asarray( vectorizer.get_feature_names() )
  for i, category in enumerate( categories ):
    top = np.argsort( classifier.coef_[i])[-n:]
    print( "%s: %s" % (category, " ".join( feature_names[ top ] ) ) )

show_top( clf, vectorizer, newsgroups_train.target_names, n=10 )
