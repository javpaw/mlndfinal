from embedding_index import get_embedding_index 
import numpy as np

def create_embedding_matrix(word_index, embedding_dim):
	vocabulary_size = len(word_index)+1
	embedding_matrix = np.zeros((vocabulary_size, embedding_dim))
	embedding_index = get_embedding_index(embedding_dim)
	for word, i in word_index.items():
		embedding_vector = embedding_index.get(word)
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	return embedding_matrix