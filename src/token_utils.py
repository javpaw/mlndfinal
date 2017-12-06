import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from datatypes import TrainingData

def create_tokenizer(train, test, vocabulary_size):
	tokenizer = Tokenizer(num_words=vocabulary_size)
	all_phrases = train.q1 + train.q2 + test.q1 + test.q2
	tokenizer.fit_on_texts(all_phrases)
	return tokenizer

def create_training_sequences(tokenizer, data, max_length):
	q1 = tokenizer.texts_to_sequences(data.q1)
	q2 = tokenizer.texts_to_sequences(data.q2)
	
	q1 = pad_sequences(q1, maxlen=max_length)
	q2 = pad_sequences(q2, maxlen=max_length)
	labels = np.array(data.labels)

	return TrainingData(q1, q2, labels)