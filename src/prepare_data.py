from get_data import load_and_clean_training_data
from get_data import load_and_clean_testing_data
from token_utils  import create_tokenizer
from token_utils import create_training_sequences
from embeddings import create_embedding_matrix
from utils import save_model_and_results
import pickle

import simple_model

TRAIN_FILE = '../data/train.csv'
TEST_FILE = '../data/test.csv'
NUM_TRAIN_SAMPLES = None;
NUM_TEST_SAMPLES = 400000;
VOCABULARY_SIZE = 200000;
MAX_WORDS_PER_QUESTION = 40
EMBEDDING_DIMENSION = 300

def checkNumSamples(exp, res, errorMessage):
  if exp != None:
    assert exp == res, errorMessage%(exp, res)


def prepare_train_data():
  print("-----------------------------Starting-----------------------------------")
  raw_train = load_and_clean_training_data(TRAIN_FILE, NUM_TRAIN_SAMPLES)
  checkNumSamples(NUM_TRAIN_SAMPLES, len(raw_train.q1), "Num train samples expected: %s, got %s")
  

  raw_test = load_and_clean_testing_data(TEST_FILE, NUM_TEST_SAMPLES)
  checkNumSamples(NUM_TEST_SAMPLES, len(raw_test.q1), "Num test samples expected: %s, got %s")


  print("-----------------------------Creating Sequences----------------------------")
  tokenizer = create_tokenizer(raw_train, raw_test, VOCABULARY_SIZE)
  train_seqs = create_training_sequences(tokenizer, raw_train, MAX_WORDS_PER_QUESTION)
  checkNumSamples(NUM_TRAIN_SAMPLES, len(train_seqs.q1), "Num train samples expected: %s, got %s")


  print("-----------------------------Creating Embedded Matrix----------------------------")
  word_index = tokenizer.word_index
  embeding_matrix = create_embedding_matrix(word_index, EMBEDDING_DIMENSION)

  expected_dimension = (len(word_index)+1, EMBEDDING_DIMENSION)
  check =  expected_dimension == embeding_matrix.shape
  error_message = "embedding dim shape doesn't fit, exp: %s, got %s"
  assert check, error_message%(expected_dimension, embeding_matrix.shape) 
  
  data = {
      "train_seqs": train_seqs,
      "embedding_matrix": embeding_matrix
  }

  pickle.dump(data, open(
    '../output_data/ready_to_train_data.p','wb'))


prepare_train_data()
