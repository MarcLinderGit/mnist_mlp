# Multi-Layer Perceptron, MNIST

In this notebook, I will train an Multi-Layer Perceptron (MLP, i.e., modern feedforward artificial neural network, consisting of fully connected neurons) to classify images from the [MNIST database](http://yann.lecun.com/exdb/mnist/) hand-written digit database.

## Project Overview

The process will be broken down into the following steps:
### 1. Load Libraries
   - Importing essential libraries including PyTorch and NumPy for data manipulation and machine learning tasks.

### 2. Load and Visualize the Data
   - Utilizing PyTorch to download and preprocess the MNIST dataset.
   - Dividing the dataset into training and testing sets, and allocating a portion for validation.
   - Creating data loaders for efficient loading of batches during training, validation, and testing.

### 3. Visualize a Batch of Training Data
   - Visualizing a batch of training images to gain insights into the nature of the data.
   - Displaying 20 images at a time, each labeled with the correct digit.

### 4. View an Image in More Detail
   - Selecting and visualizing a single image in grayscale.
   - Overlaying pixel values on the image in a 2D grid for a detailed view of intensity values.

### 5. Define the Network Architecture
   - Defining an MLP architecture named "Net" for digit classification.
   - Consists of three fully connected layers with ReLU activations and dropout layers to prevent overfitting.

### 6. Specify Loss Function and Optimizer
   - Loading necessary libraries and specifying categorical cross-entropy as the loss function.
   - Choosing stochastic gradient descent (SGD) as the optimizer with a learning rate of 0.01.

### 7. Train the Network
   - Training the MLP on the MNIST dataset for 50 epochs.
   - Monitoring and printing training and validation losses after each epoch.
   - Saving the model state if the validation loss decreases, ensuring the best-performing model is retained.

### 8. Load the Model with the Lowest Validation Loss
   - Loading the model state from the file 'model.pt,' representing the model with the lowest validation loss.

### 9. Test the Trained Network
   - Evaluating the performance of the trained MLP on the test set.
   - Calculating and printing the average test loss, test accuracy for each digit class, and overall test accuracy.

### 10. Visualize Sample Test Results
   - Visualizing a batch of test images along with their predicted and true labels.
   - Displaying images in a grid with color-coded titles (green for correct predictions, red for incorrect), providing a quick overview of the model's performance.