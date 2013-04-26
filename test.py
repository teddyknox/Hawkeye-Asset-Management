import feedparser as fp
import sqlite3
import requests
from bs4 import BeautifulSoup
import grequests

def print_res(res):
	print res.status_code

PARSE_URL = 'http://www.readability.com/api/content/v1/parser?token=c250a39459a247284924dfb275d9797082f9b420&url='
lookup_symbol = 'NASDAQ:AAPL'
url = 'https://www.google.com/finance/company_news?q=%s&num=10000' % lookup_symbol.replace(':', '%3A')
request = requests.get(url)
html_doc = request.text
soup = BeautifulSoup(html_doc, 'lxml')
article_divs = soup.find_all("div", {"class": "g-section news sfe-break-bottom-16"})
articles = []
urls = []
for article_div in article_divs:
	anchor = article_div.find('a') # get first link per item
	url = 'http:' + anchor['href']
	article = dict(
		link = url,
		title = anchor.string,
		source = article_div.find('span', {'class': 'src'}).string,
		date = article_div.find('span', {'class': 'date'}).string
	)
	urls.append(PARSE_URL + url)
	articles.append(article)
# for url in urls:
# 	response = requests.get(urls[0])
# 	if response.status_code != 200:
# 		print response.text
# 	else:
# 		print 'OK!'
# for url in urls:
	# print url

# requests = (grequests.get(urls[0], hooks=dict(response=print_res)) for u in urls)
# responses = grequests.map(requests, size=10)

# responses = grequests.map(rs)
# for response in responses:
response = requests.get(urls[0])
print response.json()

