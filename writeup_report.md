# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_image.jpg "Center Image"
[image2]: ./examples/model_loss.png "Visualization loss"

## Writeup Report

In this report I will address all steps of this project, explaining my approach and presenting some results obtained.

---
### Step 0 - Use the simulator to collect data of good driving behavior

Using the Udacity provided simulator I collected training data recorded by driving the track one. I used an analog joystick to get better measurements of the angle. I stored the data on the folder "/opt/carnd_p3/data_29_06/".

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover from getting close falling of the road.

### Step 1 - Build, a convolution neural network in Keras that predicts steering angles from images

I created the file "model.py" to train a neural network to predict steering angles from images. That file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The "model.py" file can be logically divided by 6 parts:

* Data Generator: 'generator' function (main.py line 12) which read the "center, right and left" images, adjusts the steering measurements for the side camera images, add the flipping images and steering Measurements and preprocess it on the fly, in batch size portions to feed into the convolution neural network model.
* Loading data: Load the images and steering measurements from a list of sources path into samples
* Traning and Validation Data Split: Divide the samples into trainig and validation data set using the train_test_split function from sklearn.model_selection. The split ratio used was 20% for validation size.
* Neural Network Model: Build a Convolution Neural Network using Keras. Below the model architecture is detailed.
* Train and save the model: Using the Keras function, train and save the neural network model.
* Plot Training and Validation Results: Using the Keras methods, plot the training and validation loss for each epoch.


####  Model Architecture

The strategy used to build the Convolution Neural Network was starting with a simple Neural Network (similiar with AlexNet), train it and test on simulator to verify its performance. In order to improve the model, more layers were added until achieve a model with sufficient parameters to accomplish thus problem. The final model architecture is similiar with the Nvidia pipeline.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 YUV-channel image					| 
| Cropping          	| input paramenter ((50,20), (0,0))				| 
| Lambda         		| Image normaliztion (image - 128)/128			| 
| Convolution 5x5     	| 2x2 stride, valid padding 					|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding					 	|
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding 					|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding					 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding 					|
| RELU					|												|
| Flatten				| outputs 8448  								|
| Fully connected 		| outputs 1000 									|
| Fully connected 		| outputs 100 									|
| Fully connected 		| outputs 1 									|

As it is possible to see from the table above, the model starts cropping the image by 50 row pixels from the top and 20 row pixels from the bottom of the image. Then a normalization is performed using the formula (image - 128)/128 on Keras lambda layer. Then the model has 5 convolution neural network (first 3 with 5x5 filter sizes and 2x2 stride and the last 2 with 3x3 filter sizes and 1x1 stride) with depths between 24 and 64 (model.py lines 94-112) followed by a RELU layer which introduces nonlinearity. A Flatten layer is added (model.py line 114) and is followed by 3 fully connected layers (model.py lines 116-121).


### Step 2 - Train and validate the model with a training and validation set

This is a regression problem where we want to minimize the error between the predicted steering angle and the truth meaurement angle. To do that the model has to be compiled with a regression loss function. In this case I used the mean squared error ('mse') together with the adam optimizer (model.py line 124).

During the training processs, the training and validation loss was compared to find for some possible overfitting. However, that not happened (the loss values were close). So no dropout layer was needed to add to the model.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (code line 124). The used batch size is 32 and train the model for 5 epochs.

#### Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

#### Visualization loss

For debugging purpose, I plot the training and validation loss for each epoch to check how the model performance is going. Below is an example:

![alt text][image2]

From the image we can see that model results a final training loss os around 0.013 and a last validation loss of 0.0137. So very close values which means that the model is not overfitting. 


### Step 3 - Test that the model successfully drives around track one without leaving the road

After training and saving the model as file 'model.h5', I launch the simulator on autonomous mode and run the script 'python drive.py model.h5'. I noted that the vehicle was able to drive autonomously around the track without leaving the road.
