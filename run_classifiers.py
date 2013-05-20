from ham import HAM, RandomClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
# import sklearn.ensemble.AdaBoostClassifier as abc
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

model = GaussianNB()
vectorizer = CountVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(1,3), token_pattern=ur'\b\w+\b', min_df=1)

ham = HAM(model, vectorizer)
# ham.prep_reviews_data()
ham.prep_news_data()
# ham.print_doc_feats()
# print 'Test Gaussian NB'
# ham.train_test()

# print 'Test linear SVM'	
# ham.model = SVC(kernel='linear')
# ham.train_test()

# print 'Linear SVC'
# ham.model = LinearSVC()

# print 'Test Multinomial Naive Bayes'
# ham.model = MultinomialNB()
# ham.train_test()

# print 'Test Bernoulli Naive Bayes'
# ham.model = BernoulliNB()#alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
# ham.train_test()

# print 'Test Stochastic Gradient Descent'
# ham.model = SGDClassifier(loss="hinge", penalty="l2")
# ham.train_test()

# print 'Test Gradient Boosting Classifier'
# ham.model = GradientBoostingClassifier()#n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# ham.train_test()

# print 'Test Random Forrest Classifier'
# ham.model = RandomForestClassifier(n_estimators=100)
# ham.train_test()

# print 'Test AdaBoost Classifier'
# ham.model = AdaBoostClassifier()
# ham.train_test()

# print 'Test Decision Tree Classifier'
# ham.model = DecisionTreeClassifier()
# ham.train_test()

print '\nTesting all with IdfBagofWords\n'
ham.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000, analyzer='word', ngram_range=(1, 3), token_pattern=ur'\b\w+\b', min_df=1)

print 'Test Random'
ham.model = RandomClassifier()
ham.train_test()

print 'Test Gaussian NB'
ham.model = GaussianNB()
ham.train_test()

print 'Test linear SVM'	
ham.model = SVC(kernel='linear')
ham.train_test()

print 'Test Linear SVC'
ham.model = LinearSVC()
ham.train_test()

print 'Test Multinomial Naive Bayes'
ham.model = MultinomialNB()
ham.train_test()

print 'Test Bernoulli Naive Bayes'
ham.model = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
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

# print 'Test AdaBoost Classifier'
# ham.model = AdaBoostClassifier()
# ham.train_test()

print 'Test Decision Tree Classifier'
ham.model = DecisionTreeClassifier()
ham.train_test()