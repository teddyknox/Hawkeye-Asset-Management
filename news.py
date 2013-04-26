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
		c.execute('''CREATE TABLE articles (date text, source text, symbol text, title text, content text, url text, image_url text)''')
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
			ready_articles = map(lambda a: (a['date'], a['source'], a['symbol'], a['title'], a['content'], a['url'], a['image_url']), self.articles)
			c.executemany('INSERT INTO article VALUES (?, ?, ?, ?, ?, ?, ?)', ready_articles)
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
		google_urls = []
		for article_div in article_divs:
			anchor = article_div.find('a') # get first link per item
			article = dict(
				source = article_div.find('span', {'class': 'src'}).string,
				date = article_div.find('span', {'class': 'date'}).string
			)
			google_urls.append(PARSE_URL + 'http:' + anchor['href'])
			articles.append(article)

		reqs = (grequests.get(u) for u in google_urls)
		responses = grequests.map(reqs, size=5)

		for article, res in zip(articles, responses):
			if res.status_code == 200:
				info = res.json()
				article['title'] = info['title']
				article['url'] = info['url']
				article['content'] = BeautifulSoup(info['content']).get_text()
				article['image_url'] = info['lead_image_url'] # could be cool
				article['symbol'] = lookup_symbol

		self.articles = articles

	def save_readability_parse(res):
		print res.status_code





			














