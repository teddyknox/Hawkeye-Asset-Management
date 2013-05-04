# training.py
# Training functions for HAM

from stock import *
import datetime
import math
import sqlite3

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

	def get_news_data(self):
		conn = sqlite3.connect('articles.db')
		c = conn.cursor()

		for date in self.positive_dates:
			for row in c.execute('SELECT content FROM articles WHERE symbol=? AND date=?', (self.quote, date)):
				self.positive_news_data.append(row)

		for date in self.negative_dates:
			for row in c.execute('SELECT content FROM articles WHERE symbol=? AND date=?', (self.quote, date)):
				self.negative_news_data.append(row)

	def single_word_features(self):
		for article in self.positive_news_data:
			self.positive_features.append(article.split())

		for article in self.negative_news_data:
			self.negative_features.append(article.split())

	def test_features(self, neg, pos):
		'''
		neg and pos need to be 2D arrays where each top-level array is an article,
		and each lower-level array is a feature in that article.
		'''
		

	def weight_features(self):



		
		













#t = Training("GOOG", datetime.date(2013,03,25), datetime.date(2013,03,26))
#t.get_stock_data()
#print t.stock_data

f = Training("AAPL", datetime.date(2013,03,25), datetime.date(2013,03,26))
f.positive_dates.append("2013-04-30")
f.positive_dates.append("2013-05-01")
f.get_news_data()
print f.negative_news_data
print f.positive_news_data


