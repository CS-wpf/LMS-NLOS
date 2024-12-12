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
