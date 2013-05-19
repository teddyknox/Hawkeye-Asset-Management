import pickle

def load_data(quote):
		f = open(quote+'.train', 'r')
		p = pickle.Unpickler(f)
		pos, neg, pos_prob, neg_prob = p.load()
		f.close()
		return pos, neg, pos_prob, neg_prob