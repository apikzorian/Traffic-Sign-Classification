# Traffic Sign Recognition

## Synopsis

The goal of this project is to take images of 43 different types of German traffic signs and train a neural network to classify them. After we reach an accuracy that we are comfortable with, we can test our classifier on random images of German traffic signs and see how closely we can classify them to their real categories. We will first load our dataset to visualize some of the images and understand what we are looking for. Then, we will apply some preprocessing techniques to the images to help our neural network have an easier time targeting unique aspects of the images. We will then design the architecture of our neural netowork, train it on our images, and test our model on a testing set. Finally, we will go and randomly search on the internet for 5 images of German traffic signs that were not in our data set and see just how well our network performs. Before we start, let's go over the necessary requirements to carry out this project.

## Requirements

### Packages & Libraries

- [Python 3.5](https://www.continuum.io/downloads)
- [TensorFlow](https://www.tensorflow.org/versions/)
- [numpy](https://anaconda.org/anaconda/numpy)
- [sklearn](http://scikit-learn.org/stable/install.html)
- [OpenCV](https://anaconda.org/menpo/opencv3)

### Dataset

- [Traffic Signs Data](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)

**Build a Traffic Sign Recognition Project**




## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  


## Data Set Summary & Exploration

The data set includes 3 pre-partitioned pickle files, each with its own set of images and labels. These sets constitue for our training, testing, and validation sets. The training set is, as you might have guessed, what our neural network will be trained on, while our validation set will be used to check the accuracy of our network on each run. When the network is done, it has been trained on the training set and the validation set has been bleeding into this training as well. However, our network has yet to see the testing set. The idea of keeping it completely separate from the training process is that if we want to have full confidence in our neural network, it should perform as well on a set of data is has never seen before as it did on the data is trained on. 

### Visualizing the data

[image of signs]

Before starting, we first will look at a few examples of some of the types of traffic signs we have available in our data set. Above, you can see an array of different traffic signs, with their labels above them. To get a idea of how many of each type of traffic sign we had, I graphed a histogram of the labels against the frequency of each of their traffic signs in our data:

[image of histogram]

Using `numpy.unique()`, we can get the number of unique traffic signs, as well as the number of times each one appears in our data set. 


### Splitting the data
Although we are given pre-partitioned data sets, I took the liberty of combining all of this data, shuffling it, and re-distributing the data in my own way. I personally think the training set should be significantly larger than the validation and testing set, since we want to train our network on as much data as possible. After combining the contents of the 3 pickle files and shuffling them, I used sklearn's handy 'train_test_split' function to divide my data, allocating 5% of my training data for testing, and then 20% of the remaining training data for validation


'''
X_train, X_test, y_train, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.05,
    random_state=832289)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=832289)
 '''

Using the pandas library, I was able to calculate summary statistics of the traffic signs data set:

Number of training examples = 39397
Number of validation examples = 9850
Number of testing examples = 2592
Image data shape = (32, 32, 3)
Number of classes = 43


## Image Augmentation

There are a couple things of options we can consider when pre-processing our images

### Normalizing

Normalizing our images makes it easier for our neural network to process them as input. If we do not scale our training vecotors, the ranges of our distributions of feature values would have a high variance from feature to feature. We instead want to normalize our features by subtracting the mean image value (128) and dividing by 128 again to average out the image:

'''
def pre_process_image(p_image):
    p_image = (p_image- 128.0)/128.0
    #p_image = shift_horiz_vert(p_image, 200)
    return p_image
'''

Below, we see the same set of images from before, now with normalized features

[normalized features image]

### Shifting Images

Since our final goal is to have our network be able to classify any arbitrary traffic sign, we want to take away the "pureness" of our data set. Imagine your neural network is trained on 10K images of traffic signs that were captured directly in front of the sign, so that in each image the sign is centered. First of all, this is a bad set to use, as you want to have some of these photos captured at skewed angles. To generalize our data set even more, we can shift our images horizontally and vertically. This way we can make sure that our network does not automatically always look in the center of the image to find the sign, and that it can correctly classify a sign even if parts of it do not appear in the image. Below, you can see the same random dataset from before, now randomly shifted:

[shifted image]


## Design and Test a Model Architecture



### Preprocessing


####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because ...

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 3x3	    | etc.      									|
| Fully connected		| etc.        									|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


