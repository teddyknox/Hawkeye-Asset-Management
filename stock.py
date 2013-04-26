import ystockquote


class Stock(object):

	def __init__(self, quote):
		self.quote = quote
		self.data = None
		self.percent_change = 0

	def get_percent_change(self, start_date, end_date):
		self.data = ystockquote.get_historical_prices(self.quote, convert_date(start_date), convert_date(end_date))
		days = len(self.data) - 1
		p2 = float(self.data[1][4])
		p1 = float(self.data[days][4])
		self.percent_change = (p2 - p1)/(.5 * (p1 + p2)) * 100


'''Converts dates from our standard format to '''
def convert_date(date):
	return date.replace("-","")




stock = Stock("GOOG")
stock.get_percent_change("2013-03-27", "2013-03-25")
print stock.percent_change