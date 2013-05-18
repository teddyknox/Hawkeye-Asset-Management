# Opting for a procedural approach

from util import *
from news import News
from sklearn import svm
from sklearn import naive_bayes
import math
import numpy as np
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

class HAM(object):

	def __init__(self, model, vectorizer):
		self.model = model
		self.vectorizer = vectorizer
		self.news_market_data = False

	def test_news_market(self):
		if not self.news_market_data:
			print 'Preparing corpus...'
			news = News('Resources/articles.db')
			raw = news.db_articles()
			train_raw, test_raw = divide_corpus(raw) # prep_news_data returns a tuple of vectors, labels
			self.train_vecs, self.train_labs = self.prep_news_data(train_raw, fit=True)
			self.test_vecs, self.test_labs = self.prep_news_data(test_raw)
			self.news_market_data = True

		print 'Training Model...'
		self.model.fit(self.train_vecs, self.train_labs)
		print 'Testing Model... done\n'
		return self.model.score(self.test_vecs, self.test_labs)

	def test_movie_reviews(self): # messy code to test classifier with movie reviews

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

		training_vectors = self.vectorizer.fit_transform(training, training_labels)
		testing_vectors = self.vectorizer.transform(testing)

		self.model.fit(training_vectors, training_labels)
		return self.model.score(testing_vectors, testing_labels)
		# results = test_model(model, testing_vectors, testing_labels)

	def prep_news_data(self, raw_docs, fit=False):
		docs = self.filter_old_news(raw_docs)
		doc_labels = self.news_labels(docs)
		doc_bodies = map(lambda x: x[4], docs) # 3 is title, 4 is body
		if fit:
			self.vectorizer.fit(doc_bodies, doc_labels)
		doc_vectors = self.vectorizer.transform(doc_bodies)
		if isinstance(self.model, naive_bayes.GaussianNB): # check if need dense vectors
			doc_vectors = doc_vectors.toarray()
			# doc_labels = doc_labels.toarray()			
		return doc_vectors, doc_labels

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


TRAIN_TEST_RATIO = 3 # e.g 3:1 would be 3
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
	print "Predictions: %d UP, %d DOWN" % (up_count, down_count)
	return acc

class Vectorizer(object): # Abstract class do not instantiate directly

	def fit(self, docs, labels): # should run fit_transform before transform
		self.skvectorizer.fit(docs)

	def transform(self, docs):
		return self.skvectorizer.transform(docs)
	
class BagOfWords(Vectorizer):
	def __init__(self):
		self.skvectorizer = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 2), token_pattern=ur'\b\w+\b', min_df=1)

class IdfBagOfWords(Vectorizer):
	def __init__(self):
		self.skvectorizer = CountVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 2), token_pattern=ur'\b\w+\b', min_df=1)
		self.sktransformer = TfidfTransformer()

	def fit(self, corpus, labels): # should run fit_transform before transform
		vectors = self.skvectorizer.fit(corpus, labels)
		self.sktransformer.fit(vectors, labels)

	def transform(self, corpus):
		vectors = self.skvectorizer.transform(corpus)
		return self.sktransformer.transform(vectors)
if __name__ == "__main__":
	# svm = svm.SVC(kernel='linear')
	# svm = svm.LinearSVC() 		
	gnb = naive_bayes.GaussianNB()
	bow = BagOfWords()
	ham = HAM(gnb, bow)
	acc =  ham.test_news_market()
	# vectorizer = IdfBagOfWords()
	print acc
	# print
	# print "Movie reviews data set"
	# dry_run()
	# print ham.eval_model() # run program

