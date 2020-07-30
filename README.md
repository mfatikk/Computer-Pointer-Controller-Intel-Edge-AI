# Computer Pointer Controller

Computer Pointer Controller app is used to control the movement of mouse pointer by the direction of eyes and also estimated pose of head. This app takes video as input and then app estimates eye-direction and head-pose and based on that estimation it move the mouse pointers.
I will be using the Gaze Estimation model to estimate the gaze of the user's eyes and change the mouse pointer position accordingly.
This project will demonstrate the ability to run multiple models in the same machine and coordinate the flow of data between those models.

## Demo Video of the Project

For demo video visit: https://youtu.be/KfXsvWxzthE


## Project Set Up and Installation

###Setup

####1 - Install Intel® Distribution of OpenVINO™ toolkit
- You need to install openvino successfully.
For installing openvino documentation:

- [Linux/Ubuntu](./linux-setup.md)
- [Mac](./mac-setup.md)
- [Windows](./windows-setup.md)



####2

- You need to install the prerequisites from requirements.txt using the following command.
```
pip install -r requirements.txt
```

- Initialize the OpenVINO environment 

```
cd <OpenVINO-Path>\IntelSWTools\openvino\bin\
setupvars.bat
```


####3

Download the following models by using openVINO model downloader:

**1. Face Detection Model**
```
python <OpenVINO-Path>\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py --name "face-detection-adas-binary-0001"
```
**2. Facial Landmarks Detection Model**
```
python <OpenVINO-Path>\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py --name "landmarks-regression-retail-0009"
```
**3. Head Pose Estimation Model**
```
python <OpenVINO-Path>\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py --name "head-pose-estimation-adas-0001"
```
**4. Gaze Estimation Model**
```
python <OpenVINO-Path>\IntelSWTools\openvino\deployment_tools\tools\model_downloader\downloader.py --name "gaze-estimation-adas-0002"
```

## Demo

1. Initialize the OpenVINO environment 

- Open a new terminal and run the following commands:
```
cd <OpenVINO-Path>\IntelSWTools\openvino\bin\
setupvars.bat
```

2. After initialize the OPENVINO environment, go project path.

```
cd <Project-Repo-Path>\src
```

3. Run the main.py 

```
python src/main.py -f models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -g models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i bin/demo.mp4
```
-"CPU" is the default Target device. Other devices such as "GPU", "MYRIAD", and "FPGA" is passsed based on the deployed device.
- If you want to open CAMERA, change to -i demo.mp4 to -i CAM 
- for more information run this:

```
python main.py -h
```

## Documentation

### Pre-Trained Model Documentation

* [Face Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
* [Facial Landmarks Detection Model](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
* [Head Pose Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
* [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

### Command line arguments

The *main.py* file is fed with the following arguments in the command line inference, where 

* -f "Path to Face Detection model .xml file"
* -fl "Path to Facial Landmark Detection model .xml file"
* -hp "Path to Head Pose Estimation model .xml file"
* -g "Path to Gaze Estimation model .xml file"
* -i "Path to image file, video file or CAM"
* -d "Target device"

## Benchmarks
I tested all the different performance for different model precisions on CPU and GPU devices.

###CPU

1) FP16 precision
```
python src/main.py -f models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -g models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -d CPU
```

** Model Load Time: 0.968 seconds
** Inference Time: 26.003 seconds
** FPS: 2.2689519458089813 frames/second


2) FP32 precision
```
python src/main.py -f models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -g models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -d CPU
```

** Model Load Time: 0.984 seconds
** Inference Time: 25.982 seconds
** FPS: 2.2707645132151923 frames/second

3) INT8
```
python src/main.py -f models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl models/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml -hp models/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml -g models/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -d CPU
```

** Model Load Time: 2.059 seconds
** Inference Time: 25.975 seconds
** FPS: 2.271444206618389 frames/second


###GPU

1) FP16 precision
```
python src/main.py -f models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl models/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009.xml -hp models/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001.xml -g models/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -d GPU
```

** Model Load Time: 33.100 seconds
** Inference Time: 27.334 seconds
** FPS: 2.158481050122623 frames/second


2) FP32 precision
```
python src/main.py -f models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl models/landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009.xml -hp models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml -g models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -d GPU
```

** Model Load Time: 31.844 seconds
** Inference Time: 27.105 seconds
** FPS: 2.1767340805190387 frames/second

3) INT8
```
python src/main.py -f models/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001.xml -fl models/landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009.xml -hp models/head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001.xml -g models/gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002.xml -i bin/demo.mp4 -d GPU
```

** Model Load Time: 37.122 seconds
** Inference Time: 27.382 seconds
** FPS: 2.15469009565075 frames/second


## Results
* In all precisions, GPU model loading speed is too slow. Inference Time and FPS almost same with CPU.
* If decreasing prescison, accuracy of the model decreases too.

##Edge Cases
* If there is more than one face detected, it extracts only first face and do inference on it and ignoring other faces."# Computer-Pointer-Controller-Intel-Edge-AI" 
