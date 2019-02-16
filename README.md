# Landmark Classification with Convolutional Neural Networks

A routine for classifying Istanbul's historical landmarks with a deep learning model. 

The routine is written in Python3 and Keras 2.10 on Google Colab.

## Requirements:
keras, cv2, os, numpy, matplotlib libraries and their functions are used for this routine.

## Overview:
- Downloading training materials and eliminating improper materials
- Preprocessing of materials and conversion of materials' shape into ResNet50 input shape 
- Importing ResNet50 model with imagenet pretrained weights as convolutional backbone
- Adding head layer in order to perform a custom classification
- Freezing convolutional backbone weights
- Training only head layer with 1195 landmark pictures over 5 different landmarks
- Performance evaluation with 289 landmark pictures which are not processed during training
- Unfreezing convolutional backbone weights 
- Training the whole network with 1e-3 times decreased learning rate for fine-tuning
- Performance evaluation

## Main Objects:

### Dataset: 
The dataset is provided by downloading publicly available instagram photographs with their hashtag.

Downloaded images are analized and uncorrelated images are deleted (sometimes, hashtag does not fit the images and some images are not proper materials for training.).

The dataset is preprocessed in order to create a proper input shape for ResNet50 model.

The dataset contains 1479 images with 5 different labels (Maiden's Tower, Galata Tower, Hagia Sophia, Ortakoy Mosque, Valens Aqueduct).

The dataset is divided into training and validation sets as 1189 and 289 images, respectively. 

### Model:

ResNet50 with pretrained imagenet weights is used for this project. A head layer is added to the ResNet structure for multilabel classification.

### Training:

Training is operated with the following parameters.

#### Training the head layer:

    optimizer = Stochastic Gradient Descent (SGD)
    learning_rate = 8e-3
    batch_size = 16
    class_weight = 1/(number of elements in the landmark class) for each class

#### Fine-tuning:

    optimizer = Stochastic Gradient Descent (SGD)
    learning_rate = 8e-6
    batch_size = 16
    class_weight = 1/(number of elements in the landmark class) for each class

### Performance Analysis:
In order to analyze the network' performance, 
- training and validation loss over epochs
- confusion matrix of prediction of validation set
- wrong predicted validation materials


### Results:

| training | train_loss | training_accuracy | val_loss | val_accuracy |
|------------ |------------| ------------ | ---------- | ---------- | 
| head layer | 0.1011 | 0.9849 | 0.2692 | 0.8927 |
| fine-tuning| 0.1699 | 0.9615 | 0.2042 | 0.9412 |

### Conclusion:

Model classifies five landmarks of Istanbul with %94 success rate.
