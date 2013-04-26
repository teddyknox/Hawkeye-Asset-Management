import feedparser as fp
import sqlite3
import requests
import grequests
from bs4 import BeautifulSoup
import urlparse
from time import sleep
import datetime
from sys import stdout
# from collections import namedtuple
# Article = namedtuple("Article", "date source symbol title content") # much faster lookup than a dictionary

# apple_alerts = fp.parse('http://www.google.com/alerts/feeds/16383881814015614047/13906298515147385431') # Apple Google Alert
# apple_news = fp.parse('https://www.google.com/finance/company_news?q=NASDAQ:AAPL&ei=aLx5UZCaO9GL0QHKgAE&output=rss') # Apple Google Finance
# for article in apple_alerts.entries + apple_news.entries:
# 	print article.title

READABILITY_URL = 'http://www.readability.com/api/content/v1/parser?token=c250a39459a247284924dfb275d9797082f9b420&url='

class News(object):

	def __init__(self, db_name):
		self.db_name = db_name
		self.conn = False
		self.articles = []

	def db_connect(self):
		self.conn = sqlite3.connect(self.db_name)

	def db_close(self):
		self.conn.close()
		self.conn = False

	def db_create(self):
		if not self.conn:
			self.db_connect()
		c = self.conn.cursor()
		c.execute('''CREATE TABLE IF NOT EXISTS articles (date text, source text, symbol text, title text, content text, url text, image_url text)''')
		self.conn.commit()

	def db_commit(self):
		if self.articles:
			if not self.conn:
				self.db_connect()
			c = self.conn.cursor()
			filtered_articles = filter(lambda a: a['status'] == 200, self.articles)
			ready_articles = map(lambda a: (a['date'], a['source'], a['symbol'], a['title'], a['content'], a['url'], a['image_url']), filtered_articles)
			c.executemany('INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)', ready_articles)
			self.conn.commit()
		else:
			print "No articles to commit."

	def db_articles(self, symbol):
		if not self.conn:
			self.db_connect()
		c = self.conn.cursor()
		rows = c.execute('SELECT * FROM articles WHERE symbol=?', (symbol,))
		return rows

class GoogleFinanceNews(News):
	# def __init__(self, symbol):
	# 	pass

	def parse_google_date(self, date_str):
		''' Reformats Google's dates '''
		if date_str[-3:] == 'ago': # relative date
			date = datetime.date.today().strftime('%Y-%m-%d')
		else: # string date
			date = datetime.datetime.strptime(date_str, '%b %d, %Y').strftime('%Y-%m-%d')
		return date

	def parse_readability_date(self, date_str):
		''' Removes time from date '''
		if date_str:
			return date_str[0:10]
		else:
			return False
	
	def parse_google_url(self, google_url):
		''' Extract the actual article url from the ?url= get parameter of the anchor href. '''
		get_params =  urlparse.parse_qs(urlparse.urlparse(google_url).query)
		if 'url' in get_params:
			return get_params['url'][0]
		elif 'q' in get_params:
			return get_params['q'][0]

	def scrape(self, lookup_symbol):
		url = 'https://www.google.com/finance/company_news?q=%s&num=10000' % lookup_symbol.replace(':', '%3A')
		request = requests.get(url)
		html_doc = request.text
		soup = BeautifulSoup(html_doc, 'lxml')
		article_divs = soup.find_all("div", {"class": "g-section news sfe-break-bottom-16"})

		parse_urls = []
		google_dates = []
		for i, article_div in enumerate(article_divs):
			stdout.write('\r Parsing article %d' % i)
			stdout.flush()
			google_url = article_div.find('a')['href'] # get first link per item
			parse_url = READABILITY_URL + self.parse_google_url(google_url)
			date = self.parse_google_date(article_div.find('span', {'class': 'date'}).string)

			parse_urls.append(parse_url)
			google_dates.append(date)
			# source = article_div.find('span', {'class': 'src'}).string
			# article = { date: date, source: source }
			# articles.append(article)

		responses = grequests.map((grequests.get(u) for u in parse_urls), size=10)
		# responses = []
		# for url in parse_urls:
		# 	res = requests.get(url)
		# 	responses.append(res)
			
		# 	print res.status_code
		# 	if res.status_code == 504:
		# 		print '\t' + url
		# 	sleep(.1) # throttling
	
		articles = []
		for res, google_date in zip(responses, google_dates):
			status = res.status_code

			info = res.json()
			if status == 200:
				date = self.parse_readability_date(info['date_published']) or google_date
				try:
					article = {
						'status': 		status,
						'title': 		info['title'],
						'url': 			info['url'],
						'content': 		BeautifulSoup(info['content']).get_text(),
						'image_url':	info['lead_image_url'], # could be cool
						'symbol':		lookup_symbol,
						'source': 		info['domain'],
						'date':			date,
					}
				except Exception as e:
					print e
				articles.append(article)

		self.articles = articles
