# CoND: Classification of Neuronal Differentiation

This is the code for Classification of Neuronal Differentiation using Convolutional Neural Network and Feature Analysis.
This project is carried out in cooperation with [Funahashi lab at Keio University](https://fun.bio.keio.ac.jp/).


## Overview

Our Convolutional Neural Network (CNN)-based model distinguish the neuronal differentiation and undifferentiation from the phase contrast microscopic image.


## Installation

```sh
% git clone https://gitlab.com/funalab/CoND.git
```

The code requires Python3 and the packages listed in `requirements.txt`.
The operating environment can be constructed using virtualenv as follows.

```sh
% pyenv virtualenv 3.6.3 <my/virtualenv>
% pyenv shell <my/virtualenv>
% cd <path/to/CoND>
% pyenv exec pip install -r requirements.txt
```


## QuickStart

1. Download learned model and dataset.

    - On Linux:

        ```sh
        % cd <path/to/CoND>
        % wget -P https://fun.bio.keio.ac.jp/software/CoND/best_model.npz
        % wget -P https://fun.bio.keio.ac.jp/software/CoND/dataset.zip
        % unzip dataset.zip
        ```

    - On macOS:

        ```sh
        % cd <path/to/CoND>
        % curl -O https://fun.bio.keio.ac.jp/software/CoND/best_model.npz
        % curl -O https://fun.bio.keio.ac.jp/software/CoND/dataset.zip
        % unzip dataset.zip
        ```


2. Inference on test dataset.

   The learned model using the dataset (`dataset/cross_validation/fold2`) is` best_model.npz`.
   To verify the accuracy of the learned model using test data in `dataset/cross_validation/fold2`, run the following:

   ```sh
   % python run.py --input dataset/cross_validation/fold2 --model best_model.npz [--gpu gpu]
   ```


3. Visualization of feature map by tSNE.

   Run code that visualizes the output in the input layer, third convolution layer, third pooling layer, first fully-connected layer with tSNE when all test data propagate forward.

   ```sh
   % python tSNE.py --input dataset/cross_validation/fold2 --model best_model.npz [--gpu gpu]
   ```

   We used [PredictMovingDirection](https://github.com/funalab/PredictMovingDirection) repository code for GBP and DTD feature analysis.


## How to train

Train a model with dataset `dataset/cross_validation/fold0` and performing cross validation.

```sh
% python train.py --input dataset/cross_validation/fold0 --crop_size 200 --preprocess 1 --batchsize 2 --epoch 100 [--gpu gpu]
```

The list of options will be displayed by adding `-h` option to the script.

    ```
    --indir [INDIR], -i [INDIR]                  : Specify input files directory for learning data.
    --gpu GPU, -g GPU                            : Specify GPU ID (negative value indicates CPU).
    --crop_size CROP_SIZE, -c CROP_SIZE          : Specify one side voxel size of ROI.
    --preprocess PREPROCESS, -p PREPROCESS       : Specify pre-process mode; 1. median, 2. normalization
    --batchsize BATCHSIZE, -b BATCHSIZE          : Specify minibatch size.
    --epoch EPOCH, -e EPOCH                      : Specify the number of sweeps over the dataset to train.
    ```


# Acknowledgement

The development of this algorithm was funded by a JSPS KAKENHI Grant (Number 16H04731).
