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

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[image01]: ./camera_calibration/camera_calibration_src_images/calibration1.jpg "distorted chessboard"
[image02]: ./camera_calibration/camera_calibration_calibrated_images/calibration1.jpg "undistorted chessboard"
[image03]: ./camera_calibration/camera_calibration_src_images/calibration12.jpg "distorted chessboard"
[image04]: ./camera_calibration/camera_calibration_calibrated_images/calibration12.jpg "undistorted chessboard"
[image05]: ./test_images/test1.jpg
[image06]: ./output_images/0_undistorted_images/test1.png

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

TO BE CONTINUED...

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

##### Source File: camera_calibration/camera_calibration_src_images/calibration1.jpg
Distorted Image             |  Undistorded Image
:-------------------------:|:-------------------------:
![alt text][image01]  |  ![alt text][image02]

##### Source File: camera_calibration/camera_calibration_src_images/calibration12.jpg
Distorted Image             |  Undistorded Image
:-------------------------:|:-------------------------:
![alt text][image03]  |  ![alt text][image04]

##### Source File: output_images/0_undistorted_images/test1.png
Distorted Image             |  Undistorded Image
:-------------------------:|:-------------------------:
![alt text][image05]  |  ![alt text][image06]
### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
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