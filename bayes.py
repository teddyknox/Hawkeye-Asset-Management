#bayes.py
#A simple Naive Bayes implementation for classificaiton

from load_data import *
import pickle
import sqlite3
import features
import string

class Bayes(object):

	def __init__(self, quote):
		'''
		hand in quote as a string
		'''
		self.neg_data, self.pos_data, self.pos_prob, self.neg_prob = load_data(quote)
		self.news_data = []
		self.news_features = []
		self.quote = quote
		self.date = ''

	def get_news(self):
		self.news_data = []
		self.news_features = []
		conn = sqlite3.connect('articles.db')
		c = conn.cursor()
		for row in c.execute('SELECT content FROM articles WHERE symbol=? AND date=?', (self.quote, self.date)):
			self.news_data.append(row)

	def scrub_news(self, a):
		b = a.replace("/n", '').replace('(u','')
		p = set(string.punctuation + string.digits)
		c = ''.join(ch for ch in b if ch not in p)
		d = c.replace(" u ", '')
		return d.lower()

	def single_word_features(self):
		for article in self.news_data:
			self.news_features.append(self.scrub_news(str(article)).split())

	def get_features(self):
		self.single_word_features()

		#others...

	def bayes(self):
		positive_prob_sum = 0
		negative_prob_sum = 0

		for article in self.news_features:
			p = self.pos_prob
			n = self.neg_prob
			for feature in article:
				if feature in self.pos_data:
					p *= self.pos_data[feature]
				if feature in self.neg_data:
					n *= self.neg_data[feature]
			positive_prob_sum += p
			negative_prob_sum += n
		
		return positive_prob_sum/len(self.news_features) - negative_prob_sum/len(self.news_features)

	def prob_at_date(self, date):
		'''
		hand in date as YYYY-MM-DD
		'''
		self.date = date
		self.get_news()
		self.get_features()
		b = self.bayes()

		if b < 0:
			print str(-100*b) + '% chance of NEGATIVE performance'
			return False

		if b > 0:
			print str(100*b) + '% chance of POSITIVE performance'
			return True

		if b ==0:
			print "uncertain"


#b = Bayes('AAPL')
#b.prob_at_date('2013-05-02')





		