**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./webimages/visualise_web_dataset.jpg "visualise_web_dataset"


## Rubric Points


###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to load the signnames

Number of training examples = 34799
Number of testing examples = 12630
Image data shape = (32, 32)
Number of classes = 43


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

I just plottet the images, ids and sign names. I know this signs very well so i could also verify that the labels are correct.

###Design and Test a Model Architecture

- I added a HLS colorspace to the data. 
- Did max-scaling
- I used 3 Layers conv2d_maxpool and 3 fully connected Layers
- The model did learn very fast but stuck early

- then i tried a VGG like arhitecture als so with not good enough results

- after that i modifiyed the LeNet Architecture 
- going deeper resultet in no learining at all
- going wirder did work

- after still not getting good enough results i added normalisiotion to the model input and dropout in the dense layers
  and retrained the model with succsess

####2. Training, validation and testing data

i did go with the existing data splits.

There was a lot of data, it seems it comes from videosequences moving mostly towards the signs.

I jused random batches to get rid of the sorting/sequencing.

Because of the bigger model i reduced the std_dev for the inits to 1/16.


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

    x = tf.nn.local_response_normalization(x)


My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Normalisation   		| i used local_response_normalization  							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs  14x14x6			    	|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 14x14x42.  |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 7x7x42 			     	|
| Fully connected x3 Layers		| i calculated the width depending on the input width.  	|
| Softmax				|      									|
|						|												|
|						|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

My Hyperparameters gets tuned all the time i end up with:
epochs = 100  
batch_size = 1000
keep_probability = 0.25

the epochs loop exits when the needed accuracy was hit

I tried the RMSPropOptimizer but end up with AdamOptimizer

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* Training Accuracy:     0.99400
* Validation Accuracy:   0.93243
* Testing Accuracy:      0.9409999847412109

I tryed a lot of diffrent approches
Some didn train at all, others did not reach enough accuracy
Best results i get by adopting the width of the layers my maintaining the proportions of LeNet
To avoid overfitting i retrained with normalisation at the inputlayer and increasing dropout

 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image1] 

All pictures had some difficultys
1: just signs in the background
2: overpainted
3: other text
4: small, other signs on picture, easy to confuse in grayscale
5: rotated, distorted, broken, on ground, dirty

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| slippery road     	| slippery road   								| 
| no entry  			| no entry										|
| stop					| stop											|
| traffic signals    	| general caution					 			|
| speedlimit 30       	| speedlimit 50    						     	|


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 60%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.977029383       	| slippery road   								| 
| 0.996126950 		    | no entry										|
| 0.702272236	    	| stop											|
| 0.447650850       	| general caution					 			|
| 0.797175407       	| speedlimit 50    						     	|

compleate np array:

[[  9.77029383e-01   2.19183285e-02   8.18954257e-04   1.74098066e-04    5.30776197e-05]
 [  9.96126950e-01   3.33242235e-03   4.47420840e-04   8.90796800e-05    1.77997424e-06]
 [  7.02272236e-01   1.48365945e-01   9.19106230e-02   2.69952789e-02    1.24656009e-02]
 [  4.47650850e-01   4.44780976e-01   8.56028497e-02   1.49302557e-02    5.40654548e-03]
 [  7.97175407e-01   1.86552942e-01   5.91719337e-03   3.18665686e-03    2.19294964e-03]]
    
