# Handwriting and Number Recognition Using Neural Networks

This repository contains four projects focused on handwriting and number recognition using machine learning and deep learning frameworks. Each project leverages powerful techniques such as Convolutional Neural Networks (CNNs) and TensorFlow/Keras for training and testing on datasets like MNIST.
**Project Overview**

1. **Number_Recognition_NN.ipynb**
- Implements a basic neural network for recognizing handwritten digits using TensorFlow and MNIST.
- Key Features:
  - Custom implementation of a dense neural network.
  - Uses tf.data for efficient data batching and shuffling.
  - Manual gradient computation and application using GradientTape.
  - Metrics: Loss and accuracy during training and testing.
  - Visualization of individual samples and misclassifications.
- Framework: TensorFlow.
2. **Number_Recognition_CNN.ipynb**
- Implements a Convolutional Neural Network (CNN) for recognizing handwritten digits.
- Key Features:
  - Preprocessing the MNIST dataset.
  - Building and training a custom CNN architecture.
  - Visualizing model accuracy and loss.
  - Evaluating model performance on test data.
- Framework: TensorFlow/Keras.
3. **MNIST_Keras.ipynb**
- Uses Keras' high-level API to recognize handwritten digits from the MNIST dataset.
- Key Features:
  - Simple feedforward neural network implementation.
  - Data augmentation techniques.
  - Training and testing with predefined Keras layers.
  - Visualizations of the model's predictions.
- Framework: Keras (using TensorFlow backend).
4. **Handwriting_Recognition_TF.ipynb**
- Utilizes TensorFlow to recognize and classify handwritten characters.
- Key Features:
  - Custom TensorFlow model implementation.
  - Data preprocessing and normalization.
  - Hyperparameter tuning and training visualization.
  - Model testing and evaluation.
- Framework: TensorFlow.
