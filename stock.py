import requests
import urllib
import ystockquote
from bs4 import BeautifulSoup


class Stock(object):

	def __init__(self, quote):
		self.quote = quote
		self.data = None
		self.percent_change = 0

	def get_percent_change(self, start_date, end_date):

		# q = 'select * from yahoo.finance.quote where symbol in ("'+self.quote+'")'
		q = 'select * from yahoo.finance.historicaldata where symbol = "'+self.quote+'" and startDate = "'+start_date+'" and endDate = "'+end_date+'"'
		query = urllib.quote_plus(q)
		url = "http://query.yahooapis.com/v1/public/yql?q="+query+"&env=http%3A%2F%2Fdatatables.org%2Falltables.env"
		# print url
		r = BeautifulSoup(requests.get(url).text)
		quotes = r.find_all("quote")
		# print r.prettify()
		if(len(quotes) > 0):
			p2 = float(quotes[0].close.string)
			p1 = float(quotes[1].close.string)
			self.percent_change = (p2 - p1)/(.5 * (p1 + p2)) * 100
		else:
			self.data = ystockquote.get_historical_prices(self.quote, convert_date(start_date), convert_date(end_date))
			days = len(self.data) - 1
			p2 = float(self.data[1][4])
			p1 = float(self.data[days][4])
			self.percent_change = (p2 - p1)/(.5 * (p1 + p2)) * 100



'''Converts dates from our standard format to '''
def convert_date(date):
	return date.replace("-","")



stock = Stock("GOOG")
stock.get_percent_change("2013-3-25", "2013-3-26")
print stock.percent_change
