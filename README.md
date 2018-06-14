# MNIST dataset classification using neural network in python and numpy

## MNIST
Let's begin with some intro about MNIST dataset. MNIST dataset contains grayscale images of digits from 0 - 9. These images are of dimensions 28 x 28. This dataset contains 60,000 images for training and 10,000 images for testing. Using this dataset a classifier can be trained which can take an image in the input and classify it as image of digits from 0 - 9. These images of different digits are divided into different folders named from 0 - 9. So for training these images have to be labeled with one hot encoding. The code also contains the part which converts the labels into one hot encoding.

## Network
In the code and the results provided, a very light weight neural network is used with only one layer of ten neurons. The input of network is 784 x 1 vector which is formed by reshaping a 28 x 28 image matrix into a vector. For weights update I have used stochastic gradient descent. The learning rate is set to 0.01. The output of this network is a vactor containing ten probabilities. the first probability belongs to digit 0 and the last probability is for the digit 9.

## Following is the brief description of the functions in the code

### load_dataset(path):

This function takes the path of root folder of dataset. This root folder further contains two folders called 'test' and 'train'. This function opens these folders and the subfolders in these two folders, reads the images one by one and put them in x_train or x_test. Also it stores the digit in y_train or y_test associated with the image read.

### shuffle_data(x, y):

This function takes the input values (x) and their corresponding labels (y) and shuffles them randomly.

### crossEntropyLoss(modelOutput, actualTarget):

This function takes a vector of probabilities which comes from the model output and the second parameter is the one hot encoded label corresponsing to the input example. It calculates the accumulative cross entroppy loss i.e. it sums the loss calculated for probability of every digit.

### initializeModel(numberOfLayers, inputDim, neurons):

The first argument is an integer for number of layers to be created. The second argument is also an integer for input dimension which in the case of MNIST is 784. The third argument is a list containing integers defining the number of neurons in each layer. Note that you don't need to define number of neurons for input layer. Also note that a sigmoid layer is created in the start of model. This is only to represent the input layer and is not updated during the backpropagation. Note that the number of integers in the list must be equal to the number of layers passed as argument to this function.

### sigmoid(x):

This function takes a vector and calculates the sigmoid for every value in the vector. Note that the few initial lines are added to avoid overflow or underflow because of sigmoid function.

### sigmoidGradient(activations):

This function takes the activations of a layer and calculates the derivative of sigmoid function which is simply { sigmoid * ( 1 - sigmoid ) }. Note that it is assumed that the parameter 'activations' contains values that are obtained as the output of sigmoid function.

### softmaxLossGradient(modelOutput, actualTarget):

This function is used to calculate the derivative of loss w.r.t the softmax function. This is in it's simplified form.

### forwardPropagation(model, inputSample):

This function takes the model that is being trained or already trained and does the forward propagation process using the input sample provided.

### backPropagation(model, loss, inputSample):

This function takes the model to be trained, the derivative of loss calculated after forward propagation as 'loss' and the input sample for which the loss has been calculated. This function only calculates the gradients of loss w.r.t every weight in the network except the input layer. Note that it only calculates the gradients and does not update the weights.

### weightUpdate(model, lr):

This function takes the model being trained and the learning rate as 'lr' for updating weights using gradient descent. It simply uses already calculated gradient and updates the weights of the model.

### trainModel(model, trainingData, epochs, learningRate):

This function takes the initialized model, training data, number of epochs and learning rate. It trains the model until the defined number of epochs are not being done. For each epoch it takes the training examples one by one and does the following process:

#### forward propagation
#### loss gradient calculation
#### backpropagation
#### weights update

## Results


## Acknowledgement

All of this code has been done after reading the book on the following link:
http://neuralnetworksanddeeplearning.com/

This book has been very easy and helpful in understanding the process of forward and backward propagation as well as the whole maths behind this.