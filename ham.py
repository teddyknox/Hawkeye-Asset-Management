# Opting for a procedural approach

from news import News

def article_feature_extractor(title, author, source, content):
	pass

def get_articles(symbol, date):
	return News('Resources/articles.db').db_articles(symbol)


def get_symbol_percent_change(symbol, date):
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

news = News('Resources/articles.db')
articles = news.db_articles()