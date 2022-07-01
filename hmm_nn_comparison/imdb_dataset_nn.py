import argparse, os
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras import models, layers, optimizers, losses, metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard

# constants and hyperparameters
MAX_WORD_INDEX = 10000
EMBEDDING_DIM = 128
BATCH_SIZE = 128
NUM_EPOCHS = 20
NUM_LSTM_UNITS = 32
DROPOUT_RATE = 0.2
LR = 0.001
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1.0e-8
DECAY = 0.0
VAL_PERC = 0.4


def main():
	# process input arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--model_name', type=str,
		default='model_lstm', help='--model_name=<model1|model2|...>')
	args = parser.parse_args()
	model_name	= args.model_name

	print(f'\nModel type: {model_name}')

	history_file = os.path.join(model_name, f'history_{model_name}.csv')
	logdir = os.path.join(model_name, 'log')
	ckpts = os.path.join(model_name, 'ckpts')
	os.makedirs(logdir, exist_ok=True)
	os.makedirs(ckpts, exist_ok=True)

	# load database using Keras
	(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = MAX_WORD_INDEX)

	#  print some information on the data
	max_seq_len_train = max([len(sequence) for sequence in train_data])
	max_seq_len_test = max([len(sequence) for sequence in test_data])
	min_seq_len_train = min([len(sequence) for sequence in train_data])
	min_seq_len_test = min([len(sequence) for sequence in test_data])
	print(f'Maximum train sequence length: {max_seq_len_train}')
	print(f'Maximum test sequence length: {max_seq_len_test}')
	print(f'Minimum train sequence length: {min_seq_len_train}')
	print(f'Minimum test sequence length: {min_seq_len_test}')

	# pad sequences
	X_train = keras.preprocessing.sequence.pad_sequences(train_data)
	X_test = keras.preprocessing.sequence.pad_sequences(test_data)
	print(f'X_train shape: {X_train.shape}')
	print(f'X_test shape: {X_test.shape}')

	# transform labels  into arrays
	y_train = np.asarray(train_labels).astype('float32')
	y_test  = np.asarray(test_labels).astype('float32')
	print(f'y_train shape: {y_train.shape}')
	print(f'y_test shape: {y_test.shape}')

	# build model
	model = models.Sequential()
	model.add(layers.Embedding(MAX_WORD_INDEX, EMBEDDING_DIM))
	model.add(layers.LSTM(
			units=NUM_LSTM_UNITS,
			dropout=DROPOUT_RATE,
			recurrent_dropout=DROPOUT_RATE,
			return_sequences=True
		))
	model.add(layers.LSTM(
                        units=NUM_LSTM_UNITS,
                        dropout=DROPOUT_RATE,
                        recurrent_dropout=DROPOUT_RATE
                ))
	model.add(layers.Dense(1, activation='sigmoid'))
	print(model.summary())

	# set optimizer
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

	# early stop, save best checkpoint
	filepath = ckpts + '/weights-improvement-{epoch:02d}-{val_binary_accuracy:.4f}.hdf5'
	callbacks_list = [
		EarlyStopping(
			monitor='binary_accuracy',
			patience=10),
		ModelCheckpoint(
			filepath=filepath,
			monitor='val_binary_accuracy',
			save_best_only=True,
			verbose=1),
		TensorBoard(
			log_dir=logdir),
			]

	# split training data into training and validation
	nsamples = X_train.shape[0]
	nval_samples = int(VAL_PERC * nsamples)
	X_val = X_train[:nval_samples]
	partial_X_train = X_train[nval_samples:]
	y_val = y_train[:nval_samples]
	partial_y_train = y_train[nval_samples:]

	# train model
	history = model.fit(partial_X_train,
		partial_y_train,
		epochs=NUM_EPOCHS,
		batch_size=BATCH_SIZE,
		validation_data=(X_val, y_val),
		callbacks=callbacks_list,
		verbose=1)

	# save training history
	history_df = pd.DataFrame(history.history)
	with open(history_file, mode='w') as f:
		history_df.to_csv(f)

	# score model using test data
	score, acc = model.evaluate(
			X_test, y_test,
			batch_size=BATCH_SIZE)
	print('Test score (loss):', score)
	print('Test accuracy:', acc)


if __name__ == "__main__":
	main()
