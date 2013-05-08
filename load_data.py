import pickle

def load_data(quote):
		f = open(quote+'.train', 'r')
		p = pickle.Unpickler(f)
		pos, neg = p.load()
		f.close()
		return pos, neg