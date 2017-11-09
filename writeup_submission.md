## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image01]: ./camera_calibration/camera_calibration_src_images/calibration1.jpg "distorted chessboard"
[image02]: ./camera_calibration/camera_calibration_calibrated_images/calibration1.jpg "undistorted chessboard"
[image03]: ./camera_calibration/camera_calibration_src_images/calibration12.jpg "distorted chessboard"
[image04]: ./camera_calibration/camera_calibration_calibrated_images/calibration12.jpg "undistorted chessboard"
[image05]: ./test_images/test1.jpg
[image06]: ./output_images/0_undistorted_images/test1.png
[image07]: ./test_images/test4.jpg
[image08]: ./output_images/0_undistorted_images/test4.png
[image09]: ./test_images/straight_lines1.jpg
[image10]: ./output_images/0_undistorted_images/straight_lines1.png

[image11_gray_test1]: ./output_images/1_threshold_images/grayscale_images/test1.png
[image12_gray_test4]: ./output_images/1_threshold_images/grayscale_images/test4.png
[image13_gray_straight_lines1]: ./output_images/1_threshold_images/grayscale_images/straight_lines1.png

[image14_sobelx_test1]: ./output_images/1_threshold_images/sobel_x_gray/test1.png
[image15_sobelx_test4]: ./output_images/1_threshold_images/sobel_x_gray/test4.png
[image16_sobelx_straight_lines1]: ./output_images/1_threshold_images/sobel_x_gray/straight_lines1.png

[image17_red_test1]: ./output_images/1_threshold_images/r_channel_images/test1.png
[image18_red_test4]: ./output_images/1_threshold_images/r_channel_images/test4.png
[image19_red_straight_lines1]: ./output_images/1_threshold_images/r_channel_images/straight_lines1.png

[image20_s_test1]: ./output_images/1_threshold_images/s_channel_images/test1.png
[image21_s_test4]: ./output_images/1_threshold_images/s_channel_images/test4.png
[image22_s_straight_lines1]: ./output_images/1_threshold_images/s_channel_images/straight_lines1.png

[image23_l_test1]: ./output_images/1_threshold_images/l_channel_images/test1.png
[image24_l_test4]: ./output_images/1_threshold_images/l_channel_images/test4.png
[image25_l_straight_lines1]: ./output_images/1_threshold_images/l_channel_images/straight_lines1.png

[image26_srl_test1]: ./output_images/1_threshold_images/combined_thresholds/srl_images/srl_test1.png
[image27_srl_test4]: ./output_images/1_threshold_images/combined_thresholds/srl_images/srl_test4.png
[image28_srl_straight_lines1]: ./output_images/1_threshold_images/combined_thresholds/srl_images/srl_straight_lines1.png

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup

This document is the writup submission.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.
I wrote a program 'calibrator.py' in the camera_calibration folder to calculate the camera matrix and distortion coefficients.

First, I load all filenames for the source directory location: 'camera_calibration/camera_calibration_src_images'

Second, I define parallel arrays called 'object_points' and 'image_points'.
Both sets of points represent the inner chessboard corners, 
but the image points are from the source image, and the object points are from the undistorted image.

I initialize the object point grid to be the 'ideal' chessboard corner locations.

My process for adding the object and image points was as follows:
1) I read the image using the ```cv2.imread(file_name)``` function.
2) I created a grayscale image from the rgb image using the ```cv2.cvtColor(bgr_image, cv2.COLOR_RGB2GRAY)``` function.
3) I found the chessboar corners, and whether chessboard corners were found using the following method:
```
ret, corners = cv2.findChessboardCorners(image=grayscale_image, patternSize=chessboard_dimensions)
```
-Note that ret is a boolean indicating whether corners were successfully found.
4) If corners were successfully found, I would append the corners to the image_points array
and append the object point grid to the object points array.  

-Note that these parrallel arrays are 'arrays of arrays'.

-Also note that each element in the object points array is the same object - the 'canonical' grid.

5) Then I would move on to the next file and do it again, until there were no more images.
6) Once I had collected all of the object and image points, I used the following function to calibrate the camera:
```
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, grayscale_image.shape[::-1], None, None)
```
Of use were the mtx, and dist variables.  These are the camera matrix and distortion coefficients.
7) I saved these two numpy arrays using the numpy save function:
```
np.save(file='saved_data_to_calibrate_images/mtx.npy', arr=mtx)
np.save(file='saved_data_to_calibrate_images/dist.npy', arr=dist)
```
This allowed me to load the camera matrix and distortion coefficients from other programs.

The following code is a complete method of saving the distortion coefficients.
It can also be found in camera_calibration/calibrator.py.
A test of this code can be found in camera_calibration/test_undistort.py
```python
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt

image_file_paths = glob.glob('camera_calibration_src_images/calibration*.jpg')

image_points = []  # 2D points in the image plane (from the source image)
object_points = []  # 3D points in real world space (from the destination image)

chessboard_dimensions = (9, 6)

obj_pt_grid = np.zeros((9 * 6, 3), np.float32)
obj_pt_grid[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
for path_to_file in image_file_paths:
    bgr_image = cv2.imread(path_to_file)

    # convert the image to grayscale in order to find the chessboard corners
    grayscale_image = cv2.cvtColor(bgr_image, cv2.COLOR_RGB2GRAY)
    ret, corners = cv2.findChessboardCorners(image=grayscale_image, patternSize=chessboard_dimensions)

    if ret:  # if the chessboard corners were successfully found
        image_points.append(corners)
        object_points.append(obj_pt_grid)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, grayscale_image.shape[::-1], None, None)

np.save(file='saved_data_to_calibrate_images/mtx.npy', arr=mtx)
np.save(file='saved_data_to_calibrate_images/dist.npy', arr=dist)
```

To undistort and image, I would use the following procedure:
1) Load the camera matrix and distortion coefficients.
2) Load the image and do any color channel swapping if necessary (e.g. bgr -> rgb)
3) use the ```cv2.undistort(rgb_image, mtx, dist, None, mtx)``` method, which returns the undistorted image.

The following code shows an example of undistorting an image.  This code can also be found in camera_calibration/test_undistort.py.
```
mtx = np.load('saved_data_to_calibrate_images/mtx.npy')
dist = np.load('saved_data_to_calibrate_images/dist.npy')

# load in the image files using glob
bgr_image = cv2.imread('../test_images/test1.jpg')
rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

undistorted_image = cv2.undistort(rgb_image, mtx, dist, None, mtx)
```

#### 1. Provide an example of a distortion-corrected image.
The following are some examples of undistorted images.
##### Source File: camera_calibration/camera_calibration_src_images/calibration1.jpg
Distorted Image             |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][image01]  |  ![alt text][image02]

##### Source File: camera_calibration/camera_calibration_src_images/calibration12.jpg
Distorted Image             |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][image03]  |  ![alt text][image04]

##### Source File: output_images/0_undistorted_images/test1.png
Distorted Image (test1.jpg)             |  Undistorted Image
:-------------------------:|:-------------------------:
![alt text][image05]  |  ![alt text][image06]

### Pipeline (single images)

In the begining of my pipeline, I undistort and image.  
Throughout the pipeline description, I will be using the test1.jpg, test4.jpg, and straight_lines1.jpg images.
Here is an example of it being undistorted:
##### Source File: output_images/0_undistorted_images/test1.png
Distorted Image (test1.jpg)             |  Undistorted Image (test1.jpg)|
:-------------------------:|:-------------------------:|
![alt text][image05]  |  ![alt text][image06]|

Distorted Image (test4.jpg)             |  Undistorted Image (test4.jpg)|
:-------------------------:|:-------------------------:|
![alt text][image07]  |  ![alt text][image08]|

Distorted Image (straight_lines1.jpg)             |  Undistorted Image (straight_lines1.jpg)|
:-------------------------:|:-------------------------:|
![alt text][image09]  |  ![alt text][image10]|

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I have created a image_thresholder.py file in my image_utils project to perform color transforms and gradient transforms.
I tried several color transform and gradient techniques:
* Grayscale
* Sobel X
* Red Color Channel
* S Color Channel (HLS color space)
* L Color Channel (HLS color space)
I have also tried the following combinations of color spaces:
('&' represents a bitwise and; '|' represents a bitwise or; G represents grayscale)
* S & L & R
* (S & R) | (L & G)

Original images:

|test1.jpg | test4.jpg | straight_lines.jpg |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![alt text][image06]  |  ![alt text][image08]  |  ![alt text][image10]|

Here are a list of color channels / gradients I tried:
#### 1) Grayscale

|Original test1.jpg | Grayscale test1.jpg |
|:-------------------------:|:-------------------------:|
|![alt text][image06] | ![alt text][image11_gray_test1]|
|Original test4.jpg | Grayscale test4.jpg |
|![alt text][image08] | ![alt text][image12_gray_test4]|
|Original straight_lines1.jpg | Grayscale straight_lines1.jpg |
|![alt text][image10] | ![alt text][image13_gray_straight_lines1]|

#### 2) Sobel X

|Original test1.jpg | Sobel X test1.jpg |
|:-------------------------:|:-------------------------:|
|![alt text][image06] | ![alt text][image14_sobelx_test1]|
|Original test4.jpg | Sobel X test4.jpg |
|![alt text][image08] | ![alt text][image15_sobelx_test4]|
|Original straight_lines1.jpg | Sobel X straight_lines1.jpg |
|![alt text][image10] | ![alt text][image16_sobelx_straight_lines1]|

#### 3) Red Color channel 

|Original test1.jpg | Red Channel test1.jpg |
|:-------------------------:|:-------------------------:|
|![alt text][image06] | ![alt text][image17_red_test1]|
|Original test4.jpg | Red Channel test4.jpg |
|![alt text][image08] | ![alt text][image18_red_test4]|
|Original straight_lines1.jpg | Red Channel straight_lines1.jpg |
|![alt text][image10] | ![alt text][image19_red_straight_lines1]|

#### 4) S Color channel (in HLS color space)

|Original test1.jpg | S Channel test1.jpg |
|:-------------------------:|:-------------------------:|
|![alt text][image06] | ![alt text][image20_s_test1]|
|Original test4.jpg | S Channel test4.jpg |
|![alt text][image08] | ![alt text][image21_s_test4]|
|Original straight_lines1.jpg | S Channel straight_lines1.jpg |
|![alt text][image10] | ![alt text][image22_s_straight_lines1]|

#### 5) L Color channel (in HLS color space)

|Original test1.jpg | S Channel test1.jpg |
|:-------------------------:|:-------------------------:|
|![alt text][image06] | ![alt text][image23_l_test1]|
|Original test4.jpg | S Channel test4.jpg |
|![alt text][image08] | ![alt text][image24_l_test4]|
|Original straight_lines1.jpg | S Channel straight_lines1.jpg |
|![alt text][image10] | ![alt text][image25_l_straight_lines1]|

#### 6) Combined Color Thresholds

Make some images of slr and (s&r)|(l&g)


#### Color channels compared accross images

### test1.jpg
|test1.jpg | Grayscale test1.jpg | Sobel X Channel test1.jpg |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![alt text][image06]  |  ![alt text][image11_gray_test1]  | ![alt text][image14_sobelx_test1] |
| Red Channel test1.jpg | S Channel test1.jpg | L Channel test1.jpg |
| ![alt text][image17_red_test1]| ![alt text][image20_s_test1] | ![alt text][image23_l_test1]|

### test4.jpg
|test4.jpg | Grayscale test4.jpg | Sobel X Channel test4.jpg |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![alt text][image08]  |  ![alt text][image12_gray_test4]  | ![alt text][image15_sobelx_test4] |
| Red Channel test4.jpg | S Channel test4.jpg | L Channel test4.jpg |
| ![alt text][image18_red_test4]| ![alt text][image21_s_test4] | ![alt text][image24_l_test4]|

### straight_lines1.jpg
|straight_lines1.jpg | Grayscale straight_lines1.jpg | Sobel X Channel straight_lines1.jpg |
|:-------------------------:|:-------------------------:|:-------------------------:|
|![alt text][image10]  |  ![alt text][image13_gray_straight_lines1]  | ![alt text][image16_sobelx_straight_lines1] |
| Red Channel straight_lines1.jpg | S Channel straight_lines1.jpg | L Channel straight_lines1.jpg |
| ![alt text][image19_red_straight_lines1]| ![alt text][image22_s_straight_lines1] | ![alt text][image25_l_straight_lines1]|

## Combined Colors
Each of the color channels has something to contribute and excels in some perspective.

The gray color channel excels at capturing the white lane lines, and even captures the yellow lines in ideal conditions (test1.jpg, test4.jpg, straight_lines1.jpg).
It also is moderately robust at picking up the lane lines on a bright pavement, with is a plus (test1.jpg, test4.jpg).
However, the points that it picks up can make the lane line seem faint (straight_lines1.jpg), 
and substantial noise can also be picked up (test1.jpg, test4.jpg), degrading the lane line representation.

The L color channel does a phenomenal job picking up white and yellow lines on dark pavement (straight_lines1.jpg).
But when there is bright pavement, the L color channel picks up the entire pavement (test1.jpg, test4.jpg).
This causes the L channel to become almost useless when there is bright pavement.

The red channel does a great job of picking up white and yellow lane lines on both dark and bright pavements.
It is also more robust to shadows on dark pavement (test4.jpg)
One problem with the red channels is that in addition to picking up the lane lines on bright pavement, 
it also picks up substantial noise from the bright pavement.
This adds large blobs of white around the detected lane lines in images test1.jpg and test4.jpg.

The S channel is the most robust channel.  
It does a great job at picking up yellow lane lines, and a good job of picking up white lines (all images).
It does this well on both bright pavement and dark pavement.  It also generates little noise on the pavement (all images).
The S channel does have some shortcommings, though.  
The white lane lines it picks up are faint, and not as good as the red or L channel (all images).
Also, it picks up shadows on dark pavement (test4.jpg)

#### Combining thresholds
I combined thresholds in two different ways:
* S & L & R
* (S & R) | (L & G)

I discuss the decisions for combining these thresholds and display the outputs of these thresholds.

##### S & L & R
I chose to bitwise these color channels because they seemed to be good color channels.
However, this approach mostly uses the S channel for good output.  
Anding the S with L and R mostly just eliminates some noise when there is a shadow on dark pavement.
Here are examples of the S & L & R combined channels.

|Original test1.jpg | S & L & R Channels test1.jpg |
|:-------------------------:|:-------------------------:|
|![alt text][image06] | ![alt text][image26_srl_test1]|

|Original test4.jpg | S & L & R Channels test4.jpg |
|:-------------------------:|:-------------------------:|
|![alt text][image08] | ![alt text][image27_srl_test4]|

|Original straight_lines1.jpg | S & L & R Channels straight_lines1.jpg |
|:-------------------------:|:-------------------------:|
|![alt text][image10] | ![alt text][image28_srl_straight_lines1]|

##### (S & R) | (L & G)

I ultimately wanted a result that had the robustness of the S channel while eliminating the faintness of the white lines in S, and removing the shadow issues from the S channel.
I chose to bitwise and the L and G channels together because it would capture good white lane information.
It would also eliminate the L channel picking up the hood of the car, because the gray channel does not pick up the hood.

In cases where there is a bright pavement, the L channel picks up the entire pavement, 
but the gray channel only picks up the white lines, so it is the same as having just the gray channel.

So, the benefit of bitwise anding the L and G channels is to remove some negative elements of the L channel (hood of car),
and potentially eliminate the noisyness and inconsistency of the gray channel.
Although bitwise anding these may cause a fainter line because the gray is fainter than the L.

The S channel is very robust at detecting white and yellow lanes, but activates when there is a shadow on a dark pavement (test4.jpg).
The R channel is also good at detecting white and yellow lanes, but activates when there is bright pavement (test4.jpg).
By anding these together, we can get rid of the shadow on the S channel, and the noisy bright pavement activation on the red channel.
Another benefit is that we can get rid of noise the is specific to each image, because from the picture they are noisy in different locations.

At the last step, we can or the two together: (S & R) | (L & G).
In doing this, we hope the each of the components being ored are free of noise, and that they represent mostly lane lines.
By oring them, we are taking the traits of each and combining them, so that each part (the (S&R) and the (L&G)) contributes fully.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
