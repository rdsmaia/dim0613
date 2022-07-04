import argparse, os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models, layers, optimizers, losses, metrics
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# constants and hyperparameters
MAX_WORD_INDEX = 10000
BATCH_SIZE = 128
NUM_EPOCHS = 20
LR = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1.0e-8
DECAY = 0.0
VAL_PERC = 0.4


def main():
	# process input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_path', type=str,
		required=True,
		help='--model_path=<checkpoint_path>')
	parser.add_argument('--model_name', type=str,
		default='model_lstm', help='--model_name=<model1|model2|...>')
	args = parser.parse_args()
	model_path	= args.model_path
	model_name	= args.model_name

	print(f'\nModel path: {model_path}')
	print(f'\nModel type: {model_name}')

	if model_name == 'model1':
		NUM_LSTM_UNITS = 128
		DROPOUT_RATE = 0.2
		EMBEDDING_DIM = 128
	elif model_name == 'model2':
		NUM_LSTM_UNITS = 128
		DROPOUT_RATE = 0.5
		EMBEDDING_DIM = 128
	else:
		raise ValueError(f'Model {model_name} is nor ready yet.')

	# load database using Keras
	_, (test_data, test_labels) = imdb.load_data(num_words = MAX_WORD_INDEX)

	#  print some information on the data
	max_seq_len_test = max([len(sequence) for sequence in test_data])
	min_seq_len_test = min([len(sequence) for sequence in test_data])
	print(f'Maximum test sequence length: {max_seq_len_test}')
	print(f'Minimum test sequence length: {min_seq_len_test}')

	# pad sequences
	X_test = keras.preprocessing.sequence.pad_sequences(test_data)
	print(f'X_test shape: {X_test.shape}')

	# transform labels  into arrays
	y_test  = np.asarray(test_labels).astype('float32')
	print(f'y_test shape: {y_test.shape}')

	# build model
	model = models.Sequential()
	model.add(layers.Embedding(MAX_WORD_INDEX, EMBEDDING_DIM))
	model.add(layers.LSTM(
                        units=NUM_LSTM_UNITS,
                        dropout=DROPOUT_RATE,
                        recurrent_dropout=DROPOUT_RATE
                ))
	model.add(layers.Dense(1, activation='sigmoid'))
	print(model.summary())

	# optimizer
	opt = optimizers.Adam(lr=LR,
		beta_1=BETA1,
		beta_2=BETA2,
		epsilon=EPSILON,
		decay=DECAY)

	# set loss and metrics
	loss = losses.binary_crossentropy
	met = [metrics.binary_accuracy]

	# compile model: optimization method, training criterion and metrics
	model.compile(
		optimizer=opt,
		loss=loss,
		metrics=met
	)

	# load model
	model.load_weights(model_path)

	# predict samples and obtain metrics
	y_pred = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
	y_pred_bool = np.asarray([1.0 if y_pred[i] >= 0.5 else 0.0 for i in range(y_pred.shape[0])])
	print(classification_report(y_test, y_pred_bool))


if __name__ == "__main__":
	main()
