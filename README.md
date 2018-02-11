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

### Train the Model
A convelutionary neural network is build in `train_cnn.py`. The input of the neural network is the image flow, the output is the steering angle. This is a regression problem. In the tensorflow environment, you can kick off the training by 
```python
python train_cnn.py
```

The path to the data and path to save the `h5` model is hard-coded in the `train_cnn.py` script. You will want to modify these values for your own training. 

The [Nvidia CNN architecture](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) is used in this training script. 

### Autonomous Driving

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

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



