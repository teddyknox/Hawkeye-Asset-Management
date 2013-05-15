# Opting for a procedural approach

from news import News
from sklearn import svm
import math
import sqlite3
from datetime import *
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import requests
import urllib
import ystockquote
from bs4 import BeautifulSoup

SYMBOLS = ['AAPL', 'GOOG', 'NFLX', 'TSLA', 'FB']
BAG_OF_WORDS=0
TRAIN_TEST_RATIO = 3 # e.g 3:1 would be 3

def run():
	news = News('Resources/articles.db')
	raw = news.db_articles()
	training, testing = divide_corpus(raw)

	training, training_labels = corpus_labels(training)
	testing, testing_labels = corpus_labels(testing)

	training_vectors = vectorize_corpus(training)
	testing_vectors = vectorize_corpus(testing)

	model = train_model(training_vectors, training_labels)
	results = test_model(model, testing_vectors, testing_labels)

def corpus_labels(corpus):
	''' Returns a numpy array of integer labels that correspond to the corpus docs.
	 	1 for a doc about a stock that happened to go up, -1 for a doc about a stock that went down. Removes data entry if no stock data.
	 '''
	filtered_corpus = []
	labels = []
	for doc in corpus:
		date = datetime.strptime(doc[0], '%Y-%m-%d')
		change = symbol_change(doc[2], date)
		if change: # check if we have stock data for the date
			filtered_corpus.append(doc)
			label = change/math.fabs(change) # => 1 or -1
			labels.append(label)
	return docs, np.array(labels, dtype=np.int8)

def divide_corpus(A):
	''' Takes list param and returns (bigger, smaller) according to TRAIN_TEST_RATIO. '''
	l = len(A)
	B = A[:l-l/TRAIN_TEST_RATIO]
	C = A[l-l/TRAIN_TEST_RATIO:]
	return B, C

def vectorize_corpus(corpus):
	content = map(lambda x: x[4], corpus) # just article content
	vectorizer = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 2), token_pattern=ur'\b\w+\b', min_df=1)
	return vectorizer.fit_transform(corpus)

def train_model(vectors, labels):
	model = svm.SVC()
	model.fit(vectors, labels)
	return model

def test_model(model, vectors, labels):
	preds = model.predict(data)
	correct = 0.0
	total = len(preds)
	for pred, act in zip(preds, labels):
		if pred == act:
			correct += 1
	acc = correct/total
	print "Accuracy: %d" % acc
	return acc

def convert_date(date):
	return date.replace("-", "")

def symbol_change(symbol, date):
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
	# print prevdate_str, len(r1), date_str, len(r2)
	
	# while(len(r1) == 0):
	# 	prevdate = prevdate - timedelta(days=1)
	# 	prevdate_str = prevdate.strftime('%Y-%m-%d')
	# 	r1 = list(c.execute("select price from quotes where symbol = '"+symbol+"' and date = '"+prevdate_str+"'"))

	# while(len(r2) == 0):
	# 	date = date + timedelta(days=1)
	# 	date_str = date.strftime('%Y-%m-%d')
	# 	r2 = list(c.execute("select price from quotes where symbol = '"+symbol+"' and date = '"+date_str+"'"))
	else:
		p1 = float(r1[0][0])
		p2 = float(r2[0][0])
		return (p2 - p1) / (.5 * (p1 + p2)) * 100

	# print prevdate_str, p1, date_str, p2


def symbol_change_old(symbol, date):
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

run()