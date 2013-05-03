import feedparser as fp
import sqlite3
import requests
# import grequests
from bs4 import BeautifulSoup
import urlparse
from time import sleep
import datetime
from sys import stdout
import datetime
import base64
import urllib
import pprint

# urls
READABILITY_URL = 'http://www.readability.com/api/content/v1/parser?token=c250a39459a247284924dfb275d9797082f9b420&url=%s'
REUTERS_URL = 'http://www.reuters.com/finance/stocks/companyNews?symbol=%s&date=%s'
GOOGLE_FINANCE_URL = 'https://www.google.com/finance/company_news?q=%s&num=10000'
TWITTER_SEARCH_URL = 'https://api.twitter.com/1.1/search/tweets.json'
TWITTER_PARAMS = '?&q=%s&result_type=popular&count=20&lang=en'

TWITTER_OAUTH_URL = 'https://api.twitter.com/oauth2/token'

# Twitter auth
CONSUMER_KEY = 'VVcxuEzWWyKlJ1oukYNuYw'
CONSUMER_SECRET = '4GCgk3ZSAitEb4iRZjQsLwQi9T21QJ87mROsXajxaeA'

# throttling time constants
READABILITY_THROTTLE = 0.1
TWITTER_THROTTLE = 5

class News(object):

	def __init__(self, db_name):
		self.db_name = db_name
		self.conn = False
		self.articles = []

	def db_connect(self):
		self.conn = sqlite3.connect(self.db_name)

	def readability(self, url, symbol, date):
		parse_url = READABILITY_URL % url
		response = requests.get(parse_url)
		info = response.json()
		if response.status_code == 200:
			article = {
				'title': 	info['title'],
				'url': 		info['url'],
				'content': 	BeautifulSoup(info['content']).get_text(),
				'image_url':info['lead_image_url'], # could be cool
				'symbol':	symbol,
				'source': 	info['domain'],
				'date':		self.parse_readability_date(info['date_published']) or date,
			}
			self.articles.append(article)

	def parse_readability_date(self, date_str):
		''' Removes time from date '''
		if date_str:
			return date_str[0:10]
		else:
			return False

	def db_close(self):
		self.conn.close()
		self.conn = False

	def db_create(self):
		if not self.conn:
			self.db_connect()
		c = self.conn.cursor()
		c.execute('''CREATE TABLE IF NOT EXISTS articles (date text, source text, symbol text, title text, content text, url text, image_url text, UNIQUE (title))''')
		self.conn.commit()

	def db_insert(self):
		if self.articles:
			if not self.conn:
				self.db_connect()
			c = self.conn.cursor()
			# filtered_articles = filter(lambda a: a['status'] == 200, self.articles)
			ready_articles = map(lambda a: (a['date'], a['source'], a['symbol'], a['title'], a['content'], a['url'], a['image_url']), self.articles)
			c.executemany('INSERT OR IGNORE INTO articles VALUES (?, ?, ?, ?, ?, ?, ?)', ready_articles)
			self.conn.commit()
		else:
			print "No articles to commit."

	def db_articles(self, symbol):
		if not self.conn:
			self.db_connect()
		c = self.conn.cursor()
		rows = c.execute('SELECT * FROM articles WHERE symbol=?', (symbol,))
		return rows

class TwitterNews(News):

	def authenticate(self):
		auth_string = base64.urlsafe_b64encode(CONSUMER_KEY + ':' + CONSUMER_SECRET)
		headers = {
			'Authorization': 'Basic ' + auth_string,
			'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8'
		}
		body = 'grant_type=client_credentials'
		res = requests.post(TWITTER_OAUTH_URL, data=body, headers=headers).json()
		if res['token_type'] == 'bearer':
			self.twitter_auth = {'Authorization': 'Bearer ' + res['access_token']}

	def scrape(self, symbol, usernames, keywords):

		# setup
		one_day = datetime.timedelta(days=1)
		search_date_start = datetime.datetime.now() - one_day
		search_date_end = datetime.datetime.now()
		url_set = set()
		
		total_tweets = 0
		total_articles = 0

		while True: # keep going back in time until interrupt

			# construct query string with or without keywords and usernames
			q = ''
			for i,u in enumerate(usernames):
				if i != 0:
					q += ' OR '
				q += 'from:' + u
			if q != '':
				q += ' '
			for i, k in enumerate(keywords):
				if i != 0:
					q += ' OR '
				q += k

			search_date_start_str = search_date_start.strftime("%Y-%m-%d")
			search_date_end_str = search_date_end.strftime("%Y-%m-%d")			
			search_date_start -= one_day
			search_date_end -= one_day

			q += ' since:%s until:%s' % (search_date_start_str,search_date_end_str)

			print 'Getting tweets for %s' % (search_date_end_str)

			params= TWITTER_PARAMS % urllib.quote_plus(q)
			url = TWITTER_SEARCH_URL + params
			# if until:
			# url += '&until=' + until

			response = requests.get(url, headers=self.twitter_auth).json()
			start = datetime.datetime.now()

			# this monster extracts the urls
			tweets = response.get('statuses')
			total_tweets += len(tweets)

			if tweets:
				# extract urls
				urls = map(lambda u: u[0].get('expanded_url'), filter(lambda x: bool(x), map(lambda s: s['entities']['urls'], tweets)))
				# remove repeats
				page_url_set = set(urls)
				for article_url in page_url_set:
					if article_url not in url_set:
						self.readability(article_url, symbol, search_date_start_str)
						total_articles += 1
						print '\tTweet %d, Article %d\t%s' % (total_tweets, total_articles, self.articles[-1]['title'])
						sleep(READABILITY_THROTTLE)
				url_set = url_set | page_url_set
				self.db_insert()
			else:
				print response
				print q
				break

			end = datetime.datetime.now()
			diff = (end - start).total_seconds()
			if diff < TWITTER_THROTTLE:
				sleep(TWITTER_THROTTLE - diff) # throttling for twitter


class GoogleFinanceNews(News):

	def scrape(self, lookup_symbol):
		encoded_symbol = lookup_symbol.replace(':', '%3A')
		url = GOOGLE_FINANCE_URL % encoded_symbol
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
			readability(self.parse_google_url(google_url))

	def parse_google_date(self, date_str):
		''' Reformats Google's dates '''
		if date_str[-3:] == 'ago': # relative date
			date = datetime.date.today().strftime('%Y-%m-%d')
		else: # string date
			date = datetime.datetime.strptime(date_str, '%b %d, %Y').strftime('%Y-%m-%d')
		return date
	
	def parse_google_url(self, google_url):
		''' Extract the actual article url from the ?url= get parameter of the anchor href. '''
		get_params =  urlparse.parse_qs(urlparse.urlparse(google_url).query)
		if 'url' in get_params:
			return get_params['url'][0]
		elif 'q' in get_params:
			return get_params['q'][0]
				
			# source = article_div.find('span', {'class': 'src'}).string
			# article = { date: date, source: source }
			# articles.append(article)

		# responses = grequests.map((grequests.get(u) for u in parse_urls), size=10)
		# responses = []
		# for url in parse_urls:
		# 	res = requests.get(url)
		# 	responses.append(res)
			
		# 	print res.status_code
		# 	if res.status_code == 504:
		# 		print '\t' + url
		# 	sleep(.1) # throttling
	
		# articles = []
		# for res, google_date in zip(responses, google_dates):
		# 	status = res.status_code

	
# from collections import namedtuple
# Article = namedtuple("Article", "date source symbol title content") # much faster lookup than a dictionary

# apple_alerts = fp.parse('http://www.google.com/alerts/feeds/16383881814015614047/13906298515147385431') # Apple Google Alert
# apple_news = fp.parse('https://www.google.com/finance/company_news?q=NASDAQ:AAPL&ei=aLx5UZCaO9GL0QHKgAE&output=rss') # Apple Google Finance
# for article in apple_alerts.entries + apple_news.entries:
# 	print article.title

# class ReutersNews(News):
# 	def scrape(self, symbol, date):
# 		""" date must be a datetime object """
# 		date_str = date.strftime('%m%d%Y')
# 		url = REUTERS_URL % symbol, date_str
# 		page = BeautifulSoup(requests.get(url).text, 'lxml')
# 		top_story = page.find('div', {'class': 'topStory'})
# 		top_story.find('h2')