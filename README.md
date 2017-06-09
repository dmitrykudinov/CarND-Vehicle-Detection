# Vehicle Detection Project

### The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Bonus: implement and train a Convolutional Neural Network classifier and compare performance with above SVM Classifier.
* Implement a sliding-window technique and use trained classifiers to search for vehicles in images.
* Build a pipeline to process a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/hog.png
[image3]: ./examples/bboxes_and_heat.png
[image4]: ./examples/bboxes_and_heat-cnn.png
[video1]: ./project_video_output.mp4

[Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You are reading it :)

### Histogram of Oriented Gradients (HOG)

#### Extracting HOG Features
Initial implementation using HOG features is located in "./Vehicle Detection and Tracking.ipynb".
Extraction of HOG features is implemented in code-section 2, function `get_hog_features()`.

I started by reading in all the `vehicle` and `non-vehicle` images from both GTI and KITTI datasets.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `HLS` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

I tried various combinations of parameters using the `GridSearchCV` routine: iterating through various color spaces, channels used, number of orientations. The best results I got apllying the above specified settings and using all three channels of the HLS color space.

#### Training Classifier

I trained a linear SVM in the 7th code-cell using normalized training data acquired through combining HOG, Histogram, and Color features.

Combination of all the features into a single 1D vector is hapenning in `extract_features()` function which calls all three `get_hog_features`, `bin_spatial`, and `color_hist`.

While training the SVM classifier I noticed that it's implementation is not using my GPU, replying on CPU only. This made me thinking about experimenting with another classifier built on top of a Convolutional Neural Network.

After I built one (you can fint it in "./Vehicle Detection and Tracking-CNN.ipynb") I was able to improve detection accuracy, and performance: now, using my GPU at the back, the `project_video.mp4` processing was about four times faster.

For CNN architecture I used 3 convolutional layers, spatial dropout, and then three fully connected layers all with leaky ReLU activation functions but for the very last one, where the softmax actiovation was used to predict the car/non-car choice.

### Sliding Window Search

#### Implementation
Sliding window search is implemented in 8th code-cell, `find_cars()` function.  

#### Sample images and performance optimization
I did multiple experiments applying various scales while searching for cars, trying to look for smaller size patterns the father down the road the sample is located - i.e. accounting for perspective transformation.

For example,here are some I tried:

`ystop_scale = [[656, 1.75], [600, 1.5], [575, 1.37], [563, 1.3], [555, 1.25]]`

these are bottom search boundaries, and corresponding scale values. Of course, the bigger diversity of windows to search gives the best result, but having too many windows searched had an immediate adversarial effect on the performance, so to find a reasonable compromise between quality and performance, I ended up using the following low-boundary/scale pairs:

`ystop_scale = [[656, 1.75], [600, 1.35]]`

I used originally suggested stide of two pixels, and found it performing reasonably well.

Here are some sample pictures of the sliding window algorithm, heatmaps, and thresholded heatmaps for the given test images:

![alt text][image3]


As I mentioned earlier though, I found that using a CNN classifier is a faster and more accurate option, and you can find its sliding window implementation ("./Vehicle Detection and Tracking-CNN.ipynb" `find_cars()` function) going with much bigger 0.25% (20px) overlap and producing better quality results.

Here are some sample pictures of the sliding window algorithm, heatmaps, and thresholded heatmaps for the given test images:

![alt text][image4]


### Video processing pipeline

#### Here's a [link to my video result](./project_video_output.mp4)

#### Filtering of false positives and combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video for the last 15 (in case of CNN implementation, 30 for SVM) frames.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected, and displayed number of detected blobs as a text label in the output frame.

The heatmap buffer management as well as its thresholding can be fun in the `process_image()` functions of both implementations.

---

### Discussion

1. CNN implementation performas four times faster on GPU than on CPU, but still not fast enough to keep up with real-time video stream. Options could be here: scale entire frame, not every sample (in `find_cars()` of CNN implementation).

2. CNN deals much better than SVM with shadows, bridge guard-rails producing smaller number of false positives.

3. IMHO, CNN can scale better to accommodate larger training sets including various adversarial conditions, lighting, vehicle sizes, and also distinguish between various obstacle classes like cars, pedestrians, trucks, buses, etc. without loosing so much in performance. 

So, to sum up this exercise, I consider a CNN-based approach as having a bigger potential here than SVMs. 
That was fun!

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

