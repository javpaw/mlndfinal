import pickle

def save_model_and_results(model_name, model,  history):
	h = history.history
	model.save("../output_data/%s.h5"%model_name)
	pickle.dump( h, open( "../output_data/%s-history.p"%model_name, "wb" ) )

