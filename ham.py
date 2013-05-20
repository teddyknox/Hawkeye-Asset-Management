# Opting for a procedural approach

from util import *
from news import News
from sklearn import naive_bayes
from sklearn.metrics import precision_score, recall_score, accuracy_score, classification_report
import math
from random import choice
import numpy as np
# from nltk.corpus import stopwords
# from nltk.stem.wordnet import WordNetLemmatizer

class HAM(object):

	def __init__(self, model, vectorizer):
		self.model = model
		self.vectorizer = vectorizer
		self.news_market_data = False
		self.movie_review_data = False

	def train_test(self):
		self.model.fit(self.train_vecs, self.train_labs)
		preds = self.model.predict(self.test_vecs)
		self.test_labs
		print classification_report(self.test_labs, preds, [-1,1], ['Negative','Positive'])

	def print_doc_feats(self):
		for feature in self.vectorizer.get_feature_names():
			print feature

	def prep_news_data(self):
		if not self.news_market_data:
			print 'Preparing news and stock data...\n'
			news = News('Resources/articles.db')
			raw = news.db_articles()
			train_raw, test_raw = divide_list_by_ratio(raw) # prep_news_data returns a tuple of vectors, labels
			self.train_vecs, self.train_labs = self.prep_news_articles(train_raw, fit=True)
			self.test_vecs, self.test_labs = self.prep_news_articles(test_raw)
			self.news_market_data = True
			self.movie_review_data = False

	def prep_news_articles(self, raw_docs, fit=False):
		docs = self.filter_old_news(raw_docs)
		doc_labels = self.news_labels(docs)
		doc_bodies = map(lambda x: x[4], docs) # 3 is title, 4 is body
		if fit:
			self.vectorizer.fit(doc_bodies, doc_labels)
		doc_vectors = self.vectorizer.transform(doc_bodies)
		if isinstance(self.model, naive_bayes.GaussianNB): # check if need dense vectors
			doc_vectors = doc_vectors.toarray()
		return doc_vectors, doc_labels

	def prep_reviews_data(self): # messy code to test classifier with movie reviews
		if not self.movie_review_data:
			print 'Preparing movie reviews...\n'
			from nltk.corpus import movie_reviews
			docs = [movie_reviews.raw(fileid) 
					for category in movie_reviews.categories() 
					for fileid in movie_reviews.fileids(category)]

			process = lambda x: 1 if x == 'pos' else -1
			labels = [process(category)
					for category in movie_reviews.categories() 
					for fileid in movie_reviews.fileids(category)]

			docs, labels = double_shuffle(docs, labels)
			training, testing = divide_list_by_ratio(docs)
			self.train_labs, self.test_labs = divide_list_by_ratio(labels)

			train_vecs = self.vectorizer.fit_transform(training)
			test_vecs = self.vectorizer.transform(testing)

			if isinstance(self.model, naive_bayes.GaussianNB):
				train_vecs = train_vecs.toarray()
				test_vecs = test_vecs.toarray()

			self.train_vecs = train_vecs
			self.test_vecs = test_vecs

			self.movie_review_data = True
			self.news_market_data = False		

	def filter_old_news(self, docs):
		fn = lambda d: bool(db_symbol_change(d[2], datetime.strptime(d[0], '%Y-%m-%d')))
		return filter(fn, docs)

	def news_labels(self, corpus):
		''' Returns a numpy array of integer labels that correspond to the corpus docs.
		 	1 for a doc about a stock that happened to go up, -1 for a doc about a stock that went down. Removes data entry if no stock data.
		 '''
		labels = []
		for doc in corpus:
			date = datetime.strptime(doc[0], '%Y-%m-%d')
			change = db_symbol_change(doc[2], date)
			if change:
				label = change/math.fabs(change) # => 1 or -1
				labels.append(label)
		return np.array(labels, dtype=np.int8)

	def advanced_train_test():
		self.model.fit(self.train_vecs, self.train_labs)
		preds = model.predict(self.test_vecs)
		total = len(preds)
		correct = 0.0
		up_count = 0
		down_count = 0
		for pred, act in zip(preds, self.test_labs):
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

class RandomClassifier(object):
	def fit(self, A, B):
		pass
	def predict(self, A):
		return [choice([-1, 1]) for i in range(len(A))]

# class DocPreprocessor(object):
# 	def __init__(self):
# 		self.wnl = WordNetLemmatizer()

# 	def preprocess(self, doc):
# 		def process_word(word):
# 			word = word.lower()
# 			word = self.wnl.lemmatize(word)
# 			return word
# 		return ' '.join(map(process_word, doc.split()))
