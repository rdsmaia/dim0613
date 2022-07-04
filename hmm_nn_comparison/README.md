# NN model for training and testing sentiment analysis using IMDB Movie Review Dataset.

# To train a model

$python train.py --model_name "model1|model2..."

	Example:
	$ python train.py --model_name model1

# To test the model
$python predict.py --model_name "model1|model2..." --model_path "model_checkpoint"

	Example:
	$ python evaluate.py --model_name model1 --model_path model1/ckpts/weights-improvement-03-0.8686.hdf5
