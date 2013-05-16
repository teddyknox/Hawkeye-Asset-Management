# Opting for a procedural approach

from news import News
from sklearn import svm
import math
import sqlite3
from datetime import *
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import requests
import urllib
import ystockquote
from bs4 import BeautifulSoup
from random import shuffle

SYMBOLS = ['AAPL', 'GOOG', 'NFLX', 'TSLA', 'FB']
BAG_OF_WORDS=0
TRAIN_TEST_RATIO = 3 # e.g 3:1 would be 3

def dry_run(): # messy code to test with movie reviews

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

	vectorizer = TFIDFBagOfWords()
	training_vectors = vectorizer.fit_transform(training, training_labels)
	testing_vectors = vectorizer.transform(testing)

	model = train_model(training_vectors, training_labels)
	results = test_model(model, testing_vectors, testing_labels)	

def double_shuffle(list1, list2): # shuffle two lists the same way. Used to shuffle both the labels and the docs in the dry run
	list1_shuf = []
	list2_shuf = []
	index_shuf = range(len(list1))
	shuffle(index_shuf)
	for i in index_shuf:
	    list1_shuf.append(list1[i])
	    list2_shuf.append(list2[i])
	return list1_shuf, list2_shuf

def run():
	news = News('Resources/articles.db')

	print 'Reading articles from db... ',
	raw = news.db_articles()

	print 'done\nDividing corpus... ',
	training, testing = divide_corpus(raw)

	print 'done\nAssigning stock market labels to documents... ',
	training, training_labels = corpus_labels(training)
	testing, testing_labels = corpus_labels(testing)

	print 'done\nExtracting features and vectorizing the corpus... ',
	bag_of_words = BagOfWords()
	training_vectors = bag_of_words.fit_transform(training, training_labels)
	testing_vectors = bag_of_words.transform(testing)

	print 'done\nTraining SVM... ',
	model = train_model(training_vectors, training_labels)
	print 'done\nTesting SVM... done\n'
	results = test_model(model, testing_vectors, testing_labels)


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
			print 'no stock data'
	return filtered_docs, np.array(labels, dtype=np.int8)

def divide_corpus(A):
	''' Takes list param and returns (bigger, smaller) according to TRAIN_TEST_RATIO. '''
	l = len(A)
	B = A[:l-l/TRAIN_TEST_RATIO]
	C = A[l-l/TRAIN_TEST_RATIO:]
	return B, C

def train_model(vectors, labels):
	model = svm.SVC(kernel='linear', verbose=True)
	model.fit(vectors, labels)
	return model

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
	print "Predictions: %d UP, %d DOWN" % (up_count, down_count)
	return acc

def convert_date(date):
	return date.replace("-", "")

def db_symbol_change(symbol, date):
	daynum = date.weekday()
	if daynum == 0: # if monday
		prevdate = date - timedelta(days=3)

	elif daynum == 5: # if saturday
		prevdate = date - timedelta(days=1)
		date += timedelta(days=2)

	elif daynum == 6: # if sunday
		prevdate = date - timedelta(days=2)
		date += timedelta(days=1)

	else: # if Tuesday - Friday
		prevdate = date - timedelta(days=1)

	# print "-"*30
	# print date, "====>", date.weekday()
	# print prevdate, "====>", prevdate.weekday()

	date_str = date.strftime('%Y-%m-%d')
	prevdate_str = prevdate.strftime('%Y-%m-%d')

	db = sqlite3.connect('Resources/articles.db')
	c = db.cursor()
	r1 = list(c.execute("select price from quotes where symbol = '"+symbol+"' and date = '"+prevdate_str+"'"))
	r2 = list(c.execute("select price from quotes where symbol = '"+symbol+"' and date = '"+date_str+"'"))

	db.close()

	if len(r1) == 0 or len(r2) == 0:
		return False
	else:
		p1 = float(r1[0][0])
		p2 = float(r2[0][0])
		return (p2 - p1) / (.5 * (p1 + p2)) * 100

	# print prevdate_str, p1, date_str, p2
def yahoo_symbol_change(symbol, date):
	"""Pulls data from Yahoo's API and calculates the percent change from the start data to the end date."""

	daynum = date.weekday()
	if daynum == 0: # if monday
		prevdate = date - timedelta(days=3)

	elif daynum == 5: # if saturday
		prevdate = date - timedelta(days=1)
		date += timedelta(days=2)

	elif daynum == 6: # if sunday
		prevdate = date - timedelta(days=2)
		date += timedelta(days=1)

	else: # if Tuesday - Friday
		prevdate = date - timedelta(days=1)

	date_str = date.strftime('%Y-%m-%d')
	prevdate_str = prevdate.strftime('%Y-%m-%d')

	q = 'select * from yahoo.finance.historicaldata where symbol = "%s" and startDate = "%s" and endDate = "%s"' % (symbol, date_str, prevdate_str)
	query = urllib.quote_plus(q)

	# Format URL for YQL
	url = "http://query.yahooapis.com/v1/public/yql?q=" + query + "&env=http%3A%2F%2Fdatatables.org%2Falltables.env"

	# Launch Yahoo Request
	r = BeautifulSoup(requests.get(url).text)
	symbols = r.find_all("symbol")

	# If YQL Api is not down, simply calculate percent change
	if(len(symbols) > 0): 
		p2 = float(symbols[0].close.string)
		p1 = float(symbols[1].close.string)
		return (p2 - p1) / (.5 * (p1 + p2)) * 100
	else: # Otherwise call the ystocksymbol package
		self.data = ystockquote.get_historical_prices(symbol, convert_date(date_str), convert_date(prevdate_str))
		days = len(self.data) - 1
		p2 = float(self.data[1][4])
		p1 = float(self.data[days][4])
		return (p2 - p1) / (.5 * (p1 + p2)) * 100

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

class TFIDFBagOfWords(Vectorizer):
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

run()