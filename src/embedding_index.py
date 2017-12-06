import pickle
import numpy as np
import codecs

original_path = '../data/glove.6B.%sd.txt'
picke_path = '../data/embedding_dict_%s.pickle'

def store_embedding_index(dims = 50):
	path = original_path%dims
	with open(path) as f:
		embeddings_index = {}
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs

		with open(picke_path%dims, 'wb') as store:
			pickle.dump(embeddings_index, store, pickle.HIGHEST_PROTOCOL)
 
def get_embedding_index(dims):
	with open(picke_path%dims, 'rb') as f:
			return pickle.load(f)

if __name__ == '__main__':
	#store_embedding_index(100)
	index = get_embedding_index(100)
	import ipdb; ipdb.set_trace()