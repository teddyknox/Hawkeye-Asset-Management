# Opting for a procedural approach

from news import News
from sklearn import svm
import math

news = News('Resources/articles.db')
symbols = ['AAPL', 'GOOG', 'NFLX', 'TSLA', 'FB']

def main():
	model = train()
	prediction = predict()

def train():
	docs = []
	labels = []
	for sym in symbols: # training symbols together
		start, end = news.symbol_data_date_range(sym)
		if start and end:
			date = start
			end += datetime.timedelta(days=1)
			while date.date() != end.date():
				articles = news.db_articles(sym, date)
				change = get_symbol_change(sym, date)

				label = change/math.fabs(change) # so 1 or -1
				for a in articles:
					v = article_features(a[1], a[3], a[4])
					docs.append(v)
					labels.append(label)
	model = svm.SVC()
	model.fit(docs, labels)
	return model

def predict(model, docs):
	return model.predict(docs)

def article_features(source, title, content):
	return [0,0]

def get_symbol_change(symbol, date):
	"""Pulls data from Yahoo's API and calculates the percent change from the start data to the end date."""

	daynum = date.weekday()
	if daynum == 0: # if monday
		prevdate = date - datetime.timedelta(days=3)

	if daynum == 5: # if saturday
		prevdate = date - datetime.timedelta(days=1)
		date += datetime.timedelta(days=2)

	elif daynum == 6: # if sunday
		prevdate = date - datetime.timedelta(days=2)
		date += datetime.timedelta(days=1)

	else: # if Tuesday - Friday
		prevdate = date - datetime.timedelta(days=1)

	date_str = date.strftime('%Y-%m-%d')
	prevdate_str = prevdate.strftime('%Y-%m-%d')

	q = 'select * from yahoo.finance.historicaldata where symbol = "%s" and startDate = "%s" and endDate = "%s"' % (symbol, date_str, prevdate_str)
	query = urllib.symbol_plus(q)

	# Format URL for YQL
	url = "http://query.yahooapis.com/v1/public/yql?q=" + query + "&env=http%3A%2F%2Fdatatables.org%2Falltables.env"

	# Launch Yahoo Request
	r = BeautifulSoup(requests.get(url).text)
	symbols = r.find_all("symbol")

	# If YQL Api is not down, simply calculate percent change
	if(len(symbols) > 0): 
		p2 = float(symbols[0].close.string)
		p1 = float(symbols[1].close.string)
		self.percent_change = (p2 - p1) / (.5 * (p1 + p2)) * 100
	else: # Otherwise call the ystocksymbol package
		self.data = ystocksymbol.get_historical_prices(symbol, convert_date(start_date), convert_date(end_date))
		days = len(self.data) - 1
		p2 = float(self.data[1][4])
		p1 = float(self.data[days][4])
		return (p2 - p1) / (.5 * (p1 + p2)) * 100

