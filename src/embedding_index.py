import pickle
import numpy as np
import codecs

original_path = '../data/glove.840B.300d.txt'
picke_path = '../processed_data/embedding_dict.pickle'

def store_embedding_index():
	path = original_path
	with open(path) as f:
		embeddings_index = {}
		for line in f:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs

		with open(picke_path%dims, 'wb') as store:
			pickle.dump(embeddings_index, store, pickle.HIGHEST_PROTOCOL)
 
def get_embedding_index():
	with open(picke_path, 'rb') as f:
			return pickle.load(f)