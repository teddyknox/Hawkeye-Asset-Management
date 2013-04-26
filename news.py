import feedparser as fp
import sqlite3
import requests
import grequests
from bs4 import BeautifulSoup
# from collections import namedtuple
# Article = namedtuple("Article", "date source symbol title content") # much faster lookup than a dictionary

apple_alerts = fp.parse('http://www.google.com/alerts/feeds/16383881814015614047/13906298515147385431') # Apple Google Alert
apple_news = fp.parse('https://www.google.com/finance/company_news?q=NASDAQ:AAPL&ei=aLx5UZCaO9GL0QHKgAE&output=rss') # Apple Google Finance
for article in apple_alerts.entries + apple_news.entries:
	print article.title

PARSE_URL = 'http://www.readability.com/api/content/v1/parser?token=c250a39459a247284924dfb275d9797082f9b420&url='

class News(object):

	def create_db():
		conn = sqlite3.connect('articles.db')
		c = self.conn.cursor()
		c.execute('''CREATE TABLE articles (date text, source text, symbol text, title text, content text)''')
		conn.commit()
		conn.close()

	def write_to_db():
		''' 
			Requires self.articles 
			Each article must have properties date, source, symbol, title, and content.
		'''
		if http://www.readability.com/icles:
			conn = sqlite3.connect('articles.db')
			c = self.conn.cursor()
			ready_articles = map(lambda a: (a.date, a.source, a.symbol, a.title, a.content), self.articles)
			c.executemany('INSERT INTO article VALUES (?, ?, ?, ?, ?)', ready_articles)
			conn.commit()
			conn.close()

class GoogleFinanceNews(News):
	def __init__(self, symbol):
		pass

	def scrape(self, lookup_symbol):
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
		requests = (grequests.get(urls[0], hooks=dict(response=print_res)) for u in urls)
		responses = grequests.map(requests, size=8)

	def save_readability_parse(res):
		print res.status_code





			














