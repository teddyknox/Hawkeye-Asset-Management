from news import TwitterNews

# apple = GoogleFinanceNews('Resources/articles.db')
# apple.db_create()
# apple.scrape('NASDAQ:GOOG')
# apple.db_insert()
# articles = apple.db_articles('NASDAQ:AAPL')
# for a in articles:
# 	print a[0]
# apple.db_close()

twitter = TwitterNews('Resources/articles.db')

# Twitter usernames & keywords
# hardcore hardcoding of query tuples [('SYMBOL', ['KEYWORDS', ...], ['USERNAMES', ...]), ...]
query_tuples = [('AAPL', [], []), ('GOOG', [], []),('NFLX', [], []),('TSLA', ['Tesla Motors'], []),('FB', [], [])]
twitter.scrape_wrapper(query_tuples, 1)