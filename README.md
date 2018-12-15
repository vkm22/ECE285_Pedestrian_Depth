# ECE285_Pedestrian_Depth

## Setup
Before running any code in this repository, please ensure that you have loaded the correct pod on the class cluster (launch-py3torch-gpu.sh). Our demo code relies on Python3 and CUDA 3. Then, run the following to ensure you have the appropriate versions of all Python packages:
  ```
  pip install -r requirements.txt --user
  ```
In order to run all the demo code, you will need TensorFlow 1.3.0, Keras 2.0.8, and PyTorch 0.4.0.
The Pedestrian-Detection-modified, PSMNet-modified and Mask_RCNN-modified repos contain trained models from existing repositories. In order to use code from them, you will need to run their respective setup scripts, if they exist. We only need to do this for Mask_RCNN-modified. Please issue the following commands:
```
cd Mask_RCNN-modified/
python setup.py install --user
```

## File Hierarchy
This project repository contains the following folders and their contents: 
* **Code**: This includes all of our code for this project.  
  * **Inference & Depth Estimation.ipynb**: Our main DEMO code. This gives instructions on how to run inference for Faster-RCNN, code to run a     demo of Mask R-CNN, and code to generate a depth map with PSMNet. 
  * **run_faster_rcnn_inference.sh**: Shell script that runs commands for inference on Faster-RCNN model. We cannot run this in a Jupyter         notebook, as inference simply requires running commands on the shell. The instructions to run this, however are provided in the above     main DEMO notebook.  
  * **Conversion Utilities.ipynb**: simple Python notebook to convert video to image frames with OpenCV, and vice versa
  * **depth_map_creator.py**: OpenCV Python code to generate depth maps by using the camera calibration parameters on left/right input camera images. 
  * **stereo_camera_calibration_parameters.npz**: Camera calibration parameters obtained from real measurements on the golf carts used by the UCSD Autonomous Vehicle Laboratory.
* **input_images**: input images from 2 different cameras on the golf carts used by UCSD AVL. The left front camera images are in "left_view" while the right front camera are in "right_view". These are obtained from converting the original videos to image frames at    29.97 fps. 
* **videos**: original videos collected from 2 different cameras on the golf carts on 11/9/18. 
* **RESULTS_faster_rcnn**: result images with bounding boxes obtained from running inference for Faster-RCNN, with left/right images in the     respective left_view, right_view folders. 
* **RESULTS_mask_rcnn**: result images with bounding boxes and masks obtained from running inference for Mask-RCNN. 
* **RESULTS_depth**: generated depth maps from PSMNet 
* **Mask_RCNN-modified**: This is the cloned repo (https://github.com/matterport/Mask_RCNN) that contains inference code for running Mask       RCNN. When running the demo code, the frozen inference graph (COCO weights) will be stored in this directory.
* **Pedestrian-Detection-modified**: This is the modified cloned repo (https://github.com/thatbrguy/Pedestrian-Detection) AFTER                 training/performing transfer learning. 
  * Earlier, we downloaded the train/test images from the TownCentre dataset, then downloaded the TensorFlow Object Detection API, and
    and trained this model to produce the frozen inference graph stored in Pedestrian-Detection-modified/output. For the sake of space, we     deleted the TownCentre dataset after training. 
* **PSMNet-modified**: This is the cloned repo (https://github.com/JiaRenChang/PSMNet) but with modifications. This now ONLY includes the code   snippets that we need. 
