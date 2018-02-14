# Behaviorial Cloning Project

Overview
---
This repository contains the scripts for the Behavioral Cloning Project. In this project, you can easily do the following: 

* Set up a car simulation environment to record your driving behavior
* Save the video as a set of screen shot images as the data to feed the learning agent
* A end to end steering controller using the virtual camera data on the simulator
* A conveluitonal deep neural network to clone the driving behavior

<!--This README file describes how to output the video in the "Details About Files In This Directory" section.
-->
Introduction
---
This is a course project in the [Udacity Self-Driving Car Engineering](https://github.com/udacity/CarND-Behavioral-Cloning-P3) 

### Tensorflow Environment
You can either set up the preconfigured environment according to the tuturial in [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit). Or you might want to set up your own environment. Here is is the guidlines: 

* Install the tensorflow according to the [Tutorial](https://www.tensorflow.org/install/)
* `pip install` the packages: keras, scikit-learn, socketio, eventlet, PIL, flask, opencv-python, moviepy

## Workflow

### Collect Driving Data
Download the car simulator from
* [Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae46bb_linux-sim/linux-sim.zip)
* [MacOS](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58ae4594_mac-sim.app/mac-sim.app.zip)
* [Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip)
* [Source code](https://github.com/udacity/self-driving-car-sim)
.

Open the car simulator, in the training mode, specify the directory to save the recorded images and enable recording. You might want to try driving in different scenario to extend the robustness of your controller. 

You can use the [Sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) to save time collecting the data yourself, as a start. 

### Create the Model
An 11 layer convelutionary neural network is build in [`model.py`](./model.py). You will want to first build the model before training it by

```sh
python model.py
```
The architecture of the model is designed with some modifications to the [Nvidia CNN architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). 

### Train the Model
The input of the neural network is the image flow, the output is the steering angle. This is a regression problem. In the tensorflow environment, you can kick off the training by 

```python
python train.py
```

The path to the data and path to save the `h5` model is hard-coded in the [`train.py`](./train.py). You will want to modify these values for your own training. The script is going to load the predefined `model0.h5`. Models of all epochs will be saved. 


### Autonomous Driving

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command in Python:

```python
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

Then, you can launch the car simulator you've donwloaded and run the car in the autonomous mode. 

### Saving a video of the autonomous agent (optional)

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### Convert to Videos (optional) 

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.


## References

* [https://www.youtube.com/watch?v=rpxZ87YFg0M](https://www.youtube.com/watch?v=rpxZ87YFg0M)
* [https://www.youtube.com/watch?v=EaY5QiZwSP4](https://www.youtube.com/watch?v=EaY5QiZwSP4)
* [http://selfdrivingcars.mit.edu/](http://selfdrivingcars.mit.edu/)
* [http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
* [http://jacobgil.github.io/deeplearning/vehicle-steering-angle-visualizations](http://jacobgil.github.io/deeplearning/vehicle-steering-angle-visualizations)
* [http://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c](http://medium.com/udacity/teaching-a-machine-to-steer-a-car-d73217f2492c)
* [http://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9](http://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9)
* Michael A. Nielsen, *Neural Networks and Deep Learning*, Determination Press, 2015

