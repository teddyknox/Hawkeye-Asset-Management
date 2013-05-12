import requests
import urllib
import ystockquote
from bs4 import BeautifulSoup


class Stock(object):
    """Create an instance of a stock that pulls information from YQLs API"""

    def __init__(self, quote):
        self.quote = quote
        self.data = None
        self.percent_change = 0

    def get_percent_change(self, start_date, end_date):
        """Pulls data from Yahoo's API and calculates the percent change from the start data to the end date."""
        # q = 'select * from yahoo.finance.quote where symbol in ("'
        # q += self.quote + '")'

        # Format Query for YQL
        q = 'select * from yahoo.finance.historicaldata where symbol = "'
        q += self.quote
        q += '" and startDate = "'
        q += start_date + '" and endDate = "' + end_date + '"'
        query = urllib.quote_plus(q)

        # Format URL for YQL
        url = "http://query.yahooapis.com/v1/public/yql?q="
        url += query + "&env=http%3A%2F%2Fdatatables.org%2Falltables.env"

        # Launch Yahoo Request
        r = BeautifulSoup(requests.get(url).text)
        quotes = r.find_all("quote")
        # print r.prettify()

        # If YQL Api is not down, simply calculate percent change
        if(len(quotes) > 0):
            p2 = float(quotes[0].close.string)
            p1 = float(quotes[1].close.string)
            self.percent_change = (p2 - p1) / (.5 * (p1 + p2)) * 100
        # Otherwise call the ystockquote gem
        else:
            self.data = ystockquote.get_historical_prices(self.quote, convert_date(start_date), convert_date(end_date))
            days = len(self.data) - 1
            # print self.data
            p2 = float(self.data[1][4])
            p1 = float(self.data[days][4])
            self.percent_change = (p2 - p1) / (.5 * (p1 + p2)) * 100


<<<<<<< HEAD

#stock = Stock("GOOG")
#stock.get_percent_change("2013-3-25", "2013-3-26")
#print stock.percent_change
=======
def convert_date(date):
    """Converts dates from our standard format to """
    return date.replace("-", "")
>>>>>>> f8734fc2dd2bf8c91cde019e77aff52a797b5efe
