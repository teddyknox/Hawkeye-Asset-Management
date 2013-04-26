from news import *

apple = GoogleFinanceNews('Resources/articles.db')
apple.db_create()
apple.scrape('NASDAQ:GOOG')
apple.db_insert()
articles = apple.db_articles('NASDAQ:AAPL')
for a in articles:
	print a[0]
apple.db_close()
