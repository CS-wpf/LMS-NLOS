# LMS-NLOS 
[![DOI](https://zenodo.org/badge/900532641.svg)](https://doi.org/10.5281/zenodo.14323202)

LMS-NLOS is a lightweight network with a smaller model size.

MS-NLOS imaging quality is better.

Both of these networks are passive non-line-of-sight imaging networks.

In order to avoid a significant decrease in imaging quality caused by the lightweighting of LMS-NLOS, we propose the MS-NLOS network to first improve the imaging quality of the network. 

## Dependencies
* Python 3.9
* Pytorch (2.1.0)
  Different versions may cause some errors.
* scikit-image
* opencv-python
* Tensorboard
* timm
* einops
* numpy

## Dataset

* Use the dataset collected by our laboratory for testing：

### Dataset Processing Details

At the beginning of the image collection process, a time anchor point is first set to synchronize the initial frames of all cameras; at the end of the collection, another time anchor point is set to synchronize the final frames of each camera. During the image collection process, all cameras are set to a frame rate of 25 fps, and a total of 40 video sequences are collected.

First, all data streams are converted into image data. Then, based on the initial and final frames, the data is filtered to select the image data from two cameras during the same period.

* Spatial Registration

Since non-line-of-sight images cannot have their feature points observed by the naked eye and the positions of the subjects are not fixed, only coarse registration such as cropping and scaling is performed on the VIS dataset. Manual registration is then done using the `cpselect` toolbox in Matlab, along with the `cp2tform` and `imtransform` functions.

* Temporal Registration
During data collection, although the frame rate of all cameras is set to 25 fps, to avoid discrepancies due to time errors between the cameras, temporal calibration is required when processing multi-camera data to ensure data accuracy. Since no high-precision clock synchronization equipment was used for time synchronization between all cameras, each video sequence is manually synchronized. The process is as follows:

1. First, the images from the same camera before time synchronization are processed into video using custom software.
2. Then, the video is loaded into the same timeline in Adobe Premiere Pro software, where the initial frames of all videos are aligned.
3. Frame-by-frame analysis is performed to check if the videos are synchronized. Unaligned frames are removed by frame extraction. Although this process may introduce some errors, the time discrepancy between video frames in each group is less than 5 ms after calculation.

Finally, videos with nearly identical content are obtained. The temporally synchronized videos are exported in AVI format. A program is then written to process the exported video frame by frame. For each frame, images from multiple spectral bands are cropped according to their image coordinates in the video, and the cropped images are saved separately.

Download the test dataset from the following link:

[Baidu Netdisk](https://pan.baidu.com/s/1FBUWzIGTdz736tfLNPWYfg)     
Access Code: `ict4`

* Using publicly available datasets:
1. **Download** the Anime, Supermodel, and STL-10 datasets from the link [NLOS-Passive](https://pan.baidu.com/s/19Q48BWm1aJQhIt6BF9z-uQ).
2. **Unzip** the files into the `dataset` folder.
3. **Preprocess** the dataset by running the following command:

   ```bash
   python data/preprocessing.py
   
After preparing the dataset, the folder structure should be as follows:
```
Supermodel/
├── train/
│   ├── blur/
│   │   └── (5000 image pairs)
│   │       └── xxxx.png
│   │       ...
│   └── sharp/
│       └── xxxx.png
│       ...
└── test/
    └── (1250 image pairs)
        └── (same as train)
```
## Train
The 
To train LMS-NLOS , run the command below:

```
python main.py --model_name "LMS-NLOS" --mode "train" --data_dir "dataset/Supermodel"
```

or to train MS-NLOS, run the command below:

```
python main.py --model_name "MS-NLOS" --mode "train" --data_dir "dataset/Supermodel"
```

Model weights will be saved in  results/model_name/weights folder.

## Test

* Use the dataset collected by our laboratory： 

  Use our pre-trained models and the provided test dataset to run tests.

* Using publicly available datasets:
  To test LMS-NLOS , run the command below:

  ```
  python main.py --model_name "LMS-NLOS" --mode "test" --data_dir "dataset/Supermodel" --test_model "model.pkl"
  ```

  or to test MS-NLOS, run the command below:

  ```
  python main.py --model_name "MS-NLOS" --mode "test" --data_dir "dataset/Supermodel" --test_model "model.pkl"
  ```
  
  Output images will be saved in  results/model_name/result_image folder.

## Acknowledgement

This code borrows heavily from [MIMO-UNet](https://github.com/chosj95/MIMO-UNet) and [Uformer](https://github.com/ZhendongWang6/Uformer).
