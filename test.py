<<<<<<< HEAD
from news import *

apple = GoogleFinanceNews('Resources/articles.db')
apple.db_create()
apple.scrape('NASDAQ:GOOG')
apple.db_insert()
articles = apple.db_articles('NASDAQ:AAPL')
for a in articles:
	print a[0]
apple.db_close()
=======
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
twitter.db_create()
twitter.authenticate()

# Twitter usernames & keywords
usernames = [] # ['ReutersBiz', 'FinancialTimes']
keywords = ['$AAPL']

twitter.scrape('AAPL', usernames, keywords)
>>>>>>> 69fea863a589b7b73946b7055ad18fbbb6a5f4a8
