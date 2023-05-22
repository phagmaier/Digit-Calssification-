# Digit-Calssification-
The goal of this project was to create a neural network from scratch.Used the classic MNIST digit classification training set. 
The model is incredibly simple and was created more as a challange to myself to create a NN from scartch without any outside resources or libraries (numpy doesn't count (also had to use a library for sigmoid activation function because I was getting an overload error message that I didn't know how to handle and since that wasn't part of the excerise I don't consider using an outside sigmoid function which is simple to code cheating)).
## FILES:
### main.py:
main.py should be the most sophisticated model which is why I decided to put it in a seperate file. Uses proper gradients/partial derivatives to make updates and it also incorporates momentum in an attempt to converge on a local minimum more efficiently. Supports both batches and stochastic training.
#### HOW TO RUN
build model. Can update params to chose data size iterations of your chosing. You can also specify batches which is the number of data points inside a batch. You may also specify the gradient and learning rate these are the main paramiters. 
Call train method and then evaluate the model by calling testing method. Nothing needs to be passed to any of these methods all params are passed upon initializing the model

### VARIETY
a file that contains a variety of NN of varying complexities that attempt to show the process of building up to my final chosen model. Starts with a simple stochastic model and builds up to incorporating batches, using different ways of varying complexity and accuracy to make updates and it also contains various models that do and do not update bias. Should be viewed as a file that shows stages of development. I believe the first stochasitic model is the most effective. 

## mnist_test.csv
Testing data that is the image of a digit.

## mnist_train.csv
Training data
