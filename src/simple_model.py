from keras.models import Sequential
from keras.layers import Flatten, Embedding, Dense
from sklearn.model_selection import train_test_split
import numpy as np

def create_model(train, embedding_matrix):

    vocabulary_size, embedding_dimensions = embedding_matrix.shape
    num_samples, words_per_sample =train.shape


    model = Sequential()

    embedding_layer = Embedding(vocabulary_size,
        embedding_dimensions,
        input_length=words_per_sample,
        weights = [embedding_matrix],
        trainable = False)

    # embedding_layer.set_weights([embedding_matrix])
    # embedding_layer.trainable = False

    model.add(embedding_layer)
    model.add(Flatten())

    model.add(Dense(32, activation= 'relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])

    return model

def split_data(data):
    joined_questions = np.concatenate([data.q1, data.q2], axis = 1)
    return train_test_split(joined_questions, data.labels, train_size = 0.9)
    

def run_compiled_model(model, train_seqs):
    train, val, train_labl, val_labl = train_seqs
    history = model.fit(train, train_labl, validation_data=(val, val_labl), epochs=2,
        batch_size=1024)
    return model, history

def run_model(data, embedding_matrix):
    splited_data = split_data(data)
    model = create_model(splited_data[0], embedding_matrix)
    return run_compiled_model(model, splited_data)