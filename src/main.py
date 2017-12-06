import pickle
import simple_model
from utils import save_model_and_results
from keras.models import load_model


def main():
  data =  pickle.load(open("../output_data/ready_to_train_data.p", "rb"))
  train_seqs = data["train_seqs"]
  embedding_matrix = data["embedding_matrix"]

  print("------------------------Running Simple Model----------------------------")
  model, history = simple_model.run_model(train_seqs, embedding_matrix)
  save_model_and_results('simple_model', model, history)

def get():
  model = load_model('../output_data/simple_model.h5')
  import ipdb; ipdb.set_trace()

get()