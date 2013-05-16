# Opting for a procedural approach

from util import *
from news import News
from sklearn import svm
import math
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import requests
import urllib
import ystockquote
from bs4 import BeautifulSoup

TRAIN_TEST_RATIO = 3 # e.g 3:1 would be 3

def main():
	vectorizer = TfidfBagOfWords()
	model = svm.SVC(kernel='sigmoid', verbose=True)	
	dry_run(model, vectorizer)	

def dry_run(model, vectorizer): # messy code to test with movie reviews

	from nltk.corpus import movie_reviews
	docs = [movie_reviews.raw(fileid) 
			for category in movie_reviews.categories() 
			for fileid in movie_reviews.fileids(category)]

	process = lambda x: 1 if x == 'pos' else -1
	labels = [process(category)
			for category in movie_reviews.categories() 
			for fileid in movie_reviews.fileids(category)]
	docs, labels = double_shuffle(docs, labels)
	training, testing = divide_corpus(docs)
	training_labels, testing_labels = divide_corpus(labels)

	training_vectors = vectorizer.fit_transform(training, training_labels)
	testing_vectors = vectorizer.transform(testing)

	model.fit(training_vectors, training_labels)
	print model.score(testing_vectors, testing_labels) # prints accuracy testing_vectors, testing_labels)	

def run(model, vectorizer):
	news = News('Resources/articles.db')

	raw = news.db_articles()

	training, testing = divide_corpus(raw)

	training, training_labels = corpus_labels(training)
	testing, testing_labels = corpus_labels(testing)

	training_vectors = vectorizer.fit_transform(training, training_labels)
	testing_vectors = vectorizer.transform(testing)

	model.fit(training_vectors, training_labels)
	print model.score(testing_vectors, testing_labels) # prints accuracy


def corpus_labels(corpus):
	''' Returns a numpy array of integer labels that correspond to the corpus docs.
	 	1 for a doc about a stock that happened to go up, -1 for a doc about a stock that went down. Removes data entry if no stock data.
	 '''
	filtered_docs = []
	labels = []
	for doc in corpus:
		date = datetime.strptime(doc[0], '%Y-%m-%d')
		change = db_symbol_change(doc[2], date)
		if change: # check if we have stock data for the date
			filtered_docs.append(doc)
			label = change/math.fabs(change) # => 1 or -1
			labels.append(label)
		else:
			pass # no stock
	return filtered_docs, np.array(labels, dtype=np.int8)

def divide_corpus(A):
	''' Takes list param and returns (bigger, smaller) according to TRAIN_TEST_RATIO. '''
	l = len(A)
	B = A[:l-l/TRAIN_TEST_RATIO]
	C = A[l-l/TRAIN_TEST_RATIO:]
	return B, C

def test_model(model, vectors, labels):
	preds = model.predict(vectors)
	total = len(preds)
	correct = 0.0
	up_count = 0
	down_count = 0
	for pred, act in zip(preds, labels):
		if pred == 1:
			up_count += 1
		elif pred == -1:
			down_count += 1
		if pred == act:
			correct += 1
	acc = correct/total
	print "%d/%d Correct" % (correct, total)
	print "Accuracy: %.2f" % acc
	print "Predictions: %d Positive, %d Negative" % (up_count, down_count)
	return acc

class Vectorizer(object): # Abstract class do not instantiate directly
	def pre(self, corpus):
		return map(lambda x: x[4], corpus) # just article content

	def fit_transform(self, corpus, labels): # should run fit_transform before transform
		docs = self.pre(corpus)
		return self.skvectorizer.fit_transform(docs, labels)

	def transform(self, corpus):
		docs = self.pre(corpus)
		return self.skvectorizer.transform(docs)
	
class BagOfWords(Vectorizer):
	def __init__(self):
		self.skvectorizer = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 2), token_pattern=ur'\b\w+\b', min_df=1)

class TfidfBagOfWords(Vectorizer):
	def __init__(self):
		self.skvectorizer = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 2), token_pattern=ur'\b\w+\b', min_df=1)
		self.sktransformer = TfidfTransformer()

	def fit_transform(self, corpus, labels): # should run fit_transform before transform
		docs = self.pre(corpus)
		vectors = self.skvectorizer.fit_transform(docs)
		return self.sktransformer.fit_transform(vectors, labels)

	def transform(self, corpus):
		docs = self.pre(corpus)
		vectors = self.skvectorizer.transform(docs)
		return self.sktransformer.transform(vectors)

main()