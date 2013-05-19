import requests
import re

# yfs_l84_yhoo

# yfs_p43_yhoo


def get_current_price(symbol):
    """Pulls the current stock price from Yahoo's website by scraping"""

    symbol = symbol.lower()
    url = "http://finance.yahoo.com/q?s=" + symbol + "&ql=1"

    r = requests.get(url).text
    search = "<span id=\"yfs_l84_%s\">([0-9]+\.[0-9]+)</span>" % symbol
    price = re.search(search, r)
    price = price.group(1)
    return price
