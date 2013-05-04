# training.py
# Training functions for HAM

from stock import *
import datetime
import math
import sqlite3
import string

class Training(object):

	def __init__(self, quote, start_date, end_date):
		'''
		provide dates as (YYYY, MM, DD)
		'''
		self.quote = quote
		self.start_date = start_date
		self.end_date = end_date
		self.stock = Stock(quote)
		self.stock_data = {}
		self.positive_dates = []
		self.negative_dates = []
		self.positive_news_data = []
		self.negative_news_data = []
		self.weighted_features = {}
		self.SDweight = 1   #used to change the number of standard deviations used to remove noise

	def correct_len(self, i):
		if len(i) == 1:
			i = "0" + i
		return i

	def get_stock_data(self):
		date = self.start_date

		while date < self.end_date:
			prevdate = date - datetime.timedelta(days=1)
			datestring = str(date.year) + "-" + self.correct_len(str(date.month)) + "-" + self.correct_len(str(date.day))
			prevdatestring = str(prevdate.year) + "-" + self.correct_len(str(prevdate.month)) + "-" + self.correct_len(str(prevdate.day))
			print prevdatestring, datestring
			self.stock.get_percent_change(prevdatestring, datestring)
			self.stock_data[datestring] = self.stock.percent_change
			date = date+datetime.timedelta(days=1)


	def mean(self, d):
		sum = 0
		for key in d:
			sum += d[key]
		return sum/len(d)


	def stats(self, d):
		m = self.mean(d)
		t = []
		for key in d:
			t.append((d[key] - m)**2)
		return math.sqrt(sum(t)/len(d))

	def find_extremes(self, d):
		positive = []
		negative = []
		sd = stats(d)
		m = mean(d)
		for key in d:
			d[key] = value
			if math.abs(value) - math.abs(m) > sd*self.SDweight:
				if value > m:
					positive.append(key)
				else:
					negative.append(key)
		return positive, negative

	def find_dates(self):
		self.positive_dates, self.negative_dates = find_extremes(self.stock_data)

	def scrub_news(self, a):
		b = a.replace("/n", '').replace('(u','')
		p = set(string.punctuation + string.digits)
		c = ''.join(ch for ch in b if ch not in p)
		d = c.replace(" u ", '')
		return d.lower()

	def get_news_data(self):
		conn = sqlite3.connect('articles.db')
		c = conn.cursor()

		for date in self.positive_dates:
			for row in c.execute('SELECT title FROM articles WHERE symbol=? AND date=?', (self.quote, date)):
				self.positive_news_data.append(row)

		for date in self.negative_dates:
			for row in c.execute('SELECT title FROM articles WHERE symbol=? AND date=?', (self.quote, date)):
				self.negative_news_data.append(row)


	def test_features(self, l):
		'''
		l is a 2D array where each top-level array is an article,
		and each lower-level array is a feature in that article.
		returns a dictionary keyed with the feature containing the
		frequency of that feature across the given articles.
		'''
		results = {}
		n = len(l)

		while len(l) != 0:
			article = l.pop(0)
			article_features = []
			while len(article) != 0:
				feature = article.pop(0)
				if feature not in article_features:
					article_features.append(feature)
					if feature in results:
						results[feature] += 1
					else:
						results[feature] = 1

		for key in results:
			results[key] = float(results[key])/n
		return results


	def weight_features(self, pos, neg):
		'''
		neg and pos are dictionaries keyed with features containing the
		frequency of that feature across each article set
		'''
		results = {}
		for feature in neg:
			neg[feature] *= -1

		for feature in pos:
			if feature in neg:
				results[feature] = pos[feature] + neg[feature]
				del neg[feature]
			else:
				results[feature = pos[feature]

		for feature in neg:
			results[feature] = neg[feature]

		self.weighted_features += results

	def single_word_features(self):
		pos = []
		neg = []
		for article in self.positive_news_data:
			pos.append(self.scrub_news(str(article)).split())

		for article in self.negative_news_data:
			neg.append(self.scrub_news(str(article)).split())

		p = self.test_features(pos)
		n = self.test_features(neg)

		self.weight_features(p, n)

	def all_weighted_features(self):
		'''
		include all feature tests
		'''
		self.single_word_features()

	def get_feature_lists(self):
		'''
		returns ([postive features], [negative features])
		'''
		return self.find_extremes(self.weighted_features)





#t = Training("GOOG", datetime.date(2013,03,25), datetime.date(2013,03,26))
#t.get_stock_data()
#print t.stock_data

f = Training("AAPL", datetime.date(2013,03,25), datetime.date(2013,03,26))
f.positive_dates.append("2013-04-30")
f.negative_dates.append("2013-05-01")
f.get_news_data()
pos, neg = f.single_word_features()
print pos
print "# OF ARTICLES: " + str(len(pos))
print f.test_features(pos)



