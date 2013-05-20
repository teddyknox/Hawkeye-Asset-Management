import feedparser as fp
import cpickle

fp.parse('http://www.google.com/alerts/feeds/16383881814015614047/13906298515147385431')
m = d.modified
e = d.etag

class FeedArchiver(object):

	def __init__(self, feeds=[]):
		self.feeds = feeds
		self.etags = []
		self.modified = []

	def unpickle(self, filename):
		cpickle.dump(self,filename)

	def pickle(self, pickle):

	class Feed(object):
		def __init__(self, name, url, etag=None, modified=None):
			self.name = name
			self.url = url
			self.etag = etag
			self.modified = modified
def check_all():
