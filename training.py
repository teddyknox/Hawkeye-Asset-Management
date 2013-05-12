# training.py
# Training functions for HAM

from stock import *
import datetime
import math
import sqlite3
import string
import pickle

class Training(object):

	def __init__(self, quote, start_date, end_date):
		'''
		provide dates as datetime.date(YYYY, MM, DD) object
		'''
		self.quote = quote
		self.start_date = start_date
		self.end_date = end_date
		self.stock = Stock(quote)
		self.stock_data = {}
		self.positive_dates = []
		self.negative_dates = []
		self.pos_count = 0
		self.neg_count = 0
		self.positive_news_data = []
		self.negative_news_data = []
		self.weighted_features = {}
		self.SDweight = .5   #used to change the number of standard deviations used to remove noise

	def correct_len(self, i):
		if len(i) == 1:
			i = "0" + i
		return i

	def get_stock_data(self):
		date = self.start_date

		while date < self.end_date:
			if date.weekday() == 5:
				date += datetime.timedelta(days=2)
			if date.weekday() == 6:
				date += datetime.timedelta(days=1)
			prevdate = date - datetime.timedelta(days=1)
			if prevdate.weekday() == 6:
				prevdate -= datetime.timedelta(days=2)
			datestring = str(date.year) + "-" + self.correct_len(str(date.month)) + "-" + self.correct_len(str(date.day))
			prevdatestring = str(prevdate.year) + "-" + self.correct_len(str(prevdate.month)) + "-" + self.correct_len(str(prevdate.day))
			# print prevdatestring, datestring  #for debugging 
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
		sd = self.stats(d)
		m = self.mean(d)
		for key in d:
			value = d[key]
			if math.fabs(value - m) > sd*self.SDweight:
				if value > m:
					positive.append(key)
				else:
					negative.append(key)
		return positive, negative

	def find_dates(self):
		self.positive_dates, self.negative_dates = self.find_extremes(self.stock_data)


	def scrub_news(self, a):
		b = a.replace("\n", '').replace('(u','')
		p = set(string.punctuation + string.digits)
		c = ''.join(ch for ch in b if ch not in p)
		d = c.replace(" u ", '')
		return d.lower()

	def get_news_data(self):
		conn = sqlite3.connect('articles.db')
		c = conn.cursor()

		for date in self.positive_dates:
			for row in c.execute('SELECT content FROM articles WHERE symbol=? AND date=?', (self.quote, date)):
				self.positive_news_data.append(row)

		for date in self.negative_dates:
			for row in c.execute('SELECT content FROM articles WHERE symbol=? AND date=?', (self.quote, date)):
				self.negative_news_data.append(row)

		self.pos_count, self.neg_count = len(self.positive_news_data), len(self.negative_news_data)


	def test_features(self, l):
		'''
		l is an array containing arrays of article features.
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
				results[feature] = pos[feature]

		for feature in neg:
			results[feature] = neg[feature]

		self.weighted_features = dict(results.items() + self.weighted_features.items())

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

		#others...

	def get_feature_lists(self):
		'''
		returns ([postive features], [negative features])
		'''
		pos_list, neg_list = self.find_extremes(self.weighted_features)

		p = {}
		n = {}

		for f in pos_list:
			p[f] = self.weighted_features[f]

		for f in neg_list:
			n[f] = self.weighted_features[f] * -1


		return p,n


	def train_to_pickle(self):
		'''
		does it all and spits it into a file named [index].train
		'''
		self.get_stock_data() 
		self.find_dates() 

		#self.positive_dates.append("2013-04-30")  #used for testing 
		#self.negative_dates.append("2013-05-01")  #while stocks are down

		self.get_news_data()
		self.all_weighted_features()

		t_count = self.pos_count + self.neg_count

		#save to pickle
		f = open(self.quote+'.train', 'w')
		p = pickle.Pickler(f)

		pos, neg = self.get_feature_lists()
		p.dump((pos, neg, float(self.pos_count)/t_count, float(self.neg_count)/t_count))
		f.close()







#t = Training("GOOG", datetime.date(2013,03,25), datetime.date(2013,03,26))
#t.get_stock_data()
#print t.stock_data

#f = Training("AAPL", datetime.date(2013,04,24), datetime.date(2013,05,03))
#f.train_to_pickle()
#f.positive_dates.append("2013-04-30")
#f.negative_dates.append("2013-05-01")
#f.get_news_data()




