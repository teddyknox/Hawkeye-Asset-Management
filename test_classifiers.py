#test classifiers.py
#tests for a given range 
from training import *
import datetime
from stock import *

from bayes import *
#others...

def date_string(date):
	'''
	helper method for changing datetime objects into strings
	'''
	year = str(date.year)
	month = str(date.month)
	day = str(date.day)

	if len(month) == 1:
		month = "0" + month

	if len(day) == 1:
		day = "0" + day
	
	return year + "-" + month + "-" + day

def test_classifiers(quote, training_date_1, training_date_2, test_date_1, test_date_2):
	bayes_successes = []
	#others ..

	t = Training(quote, training_date_1, training_date_2)
	s = Stock(quote)
	t.train_to_pickle()
	b = Bayes(quote)
	#others..

	date = test_date_1

	while date <= test_date_2:
		bayes_result = b.prob_at_date(date_string(date))
		#others..

		if date.weekday() != 5 and date.weekday() != 6 and date < date.today():
			delta = 1
			if date.weekday() == 0:
				delta = 3
			actual = s.get_percent_change(date_string(date - datetime.timedelta(days=delta)), date_string(date))
			up = True
			if actual < 0:
				up = False
			bayes_successes.append(bayes_result == up)
		date += datetime.timedelta(days=1)

	bayes_accuracy = bayes_successes.count(True)/len(bayes_successes)
	bayes_count = len(bayes_successes)

	print "bayes returned " + str(bayes_accuracy) + " successful results with " + str(bayes_count) + " trials."

	return float(bayes_accuracy)/bayes_count

def test_all_stocks(training_date_1, training_date_2, test_date_1, test_date_2):
	conn = sqlite3.connect('articles.db')
	c = conn.cursor()
	symbols = c.execute('SELECT DISTINCT symbol FROM articles')

	accuracies = []

	for symbol in symbols:
		s = str(symbol).replace("(u'",'').replace("',)",'')
		print s
		result = test_classifiers(s, training_date_1, training_date_2, test_date_1, test_date_2)
		accuracies.append(result)

	print "Overall average accuracy of " + str(sum(accuracies)/len(accuracies)*100) + "%"

test_all_stocks(datetime.date(2013,04,24), datetime.date(2013,05,02), datetime.date(2013,05,02), datetime.date(2013,05,03))

#test_classifiers('AAPL', datetime.date(2013,04,24), datetime.date(2013,05,02), datetime.date(2013,05,02), datetime.date(2013,05,03))






