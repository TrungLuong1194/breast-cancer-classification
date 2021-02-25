# Deep-learning-project

## Link dataset

https://www.kaggle.com/paultimothymooney/breast-histopathology-images

## Structure directory

- main
	- CNN.ipynb
	- CapsNet.ipynb
	- MLP.ipynb
	- Process_Data.ipynb
- dataset
	- 8863/[0-1]
	- 10300/[0-1]
	- ...
- data_batch_train
	- X_train_1.hdf5
	- ...
- data_batch_test
	- X_test_1.hdf5
	- ...
- data_batch_valid
	- X_valid_1.hdf5
	- ...

## Algorithms applied

- Multilayer perceptron

https://en.wikipedia.org/wiki/Multilayer_perceptron

- Convolutional neural network

https://en.wikipedia.org/wiki/Convolutional_neural_network

- Capsule neural network

https://en.wikipedia.org/wiki/Capsule_neural_network

## Result

|              | MLP    | CNN    | CapsNet |
| Accuracy     | 84.25% | 88.52% | 87.41%  |
| F1 score     | 0.70   | 0.81   | 0.79    |
| Recall score | 0.64   | 0.86   | 0.81    |
