from ham import HAM
from sklearn import svm
from sklearn import naive_bayes
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

model = naive_bayes.GaussianNB()
vectorizer = CountVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(2,3), token_pattern=ur'\b\w+\b', min_df=1)

ham = HAM(model, vectorizer)
ham.prep_reviews_data()
# ham.prep_news_data()

print 'Test Gaussian NB'
ham.train_test()

print 'Test linear SVM'	
ham.model = svm.SVC(kernel='linear')
ham.train_test()

print 'Linear SVC'
ham.model = svm.LinearSVC()

print 'Test Multinomial Naive Bayes'
ham.model = naive_bayes.MultinomialNB()
ham.train_test()

print 'Test Bernoulli Naive Bayes'
ham.model = naive_bayes.BernoulliNB()#alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
ham.train_test()

print 'Test Stochastic Gradient Descent'
ham.model = SGDClassifier(loss="hinge", penalty="l2")
ham.train_test()

print 'Test Gradient Boosting Classifier'
ham.model = GradientBoostingClassifier()#n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
ham.train_test()

print 'Test AdaBoost Classifier'
ham.model = RandomForestClassifier(n_estimators=100)
ham.train_test()

print '\nTesting all with IdfBagofWords\n'
ham.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(2, 3), token_pattern=ur'\b\w+\b', min_df=1)

print 'Test Gaussian NB'
ham.model = naive_bayes.GaussianNB()
ham.train_test()

print 'Test linear SVM'	
ham.model = svm.SVC(kernel='linear')
ham.train_test()

print 'Test Linear SVC'
ham.model = svm.LinearSVC()
ham.train_test()

print 'Test Multinomial Naive Bayes'
ham.model = naive_bayes.MultinomialNB()
ham.train_test()

print 'Test Bernoulli Naive Bayes'
ham.model = naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
ham.train_test()

print 'Test Stochastic Gradient Descent'
ham.model = SGDClassifier(loss="hinge", penalty="l2")
ham.train_test()

print 'Test Gradient Boosting Classifier'
ham.model = GradientBoostingClassifier()#n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
ham.train_test()

print 'Test Random Forrest Classifier'
ham.model = RandomForestClassifier(n_estimators=100)
ham.train_test()

print 'Test AdaBoost Classifier'
ham.model = RandomForestClassifier(n_estimators=100)
ham.train_test()	