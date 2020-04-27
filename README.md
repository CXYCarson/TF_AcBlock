# TF2 reimplementation of Asymmetric Convolution Module

This is a re-implementation of the paper: 'ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks'

Original paper site: https://arxiv.org/abs/1908.03930

Original Github site(Pytorch): https://github.com/DingXiaoH/ACNet

Currently only supports tf.keras API, does not support other TensorFlow APIs such as tf.layers, but you can look into the code and figure out ways around.

### Requirement 

- Tensorflow 2
- numpy
    
### How to use

1. Clone into your local directory.  
    ```
    git clone https://github.com/CXYCarson/TF_AcBlock.git
    ```
2. To train a cifar-quick without AC blocks, run
    ```
    python create_cfqk.py
    ```
3. To train a cifar-quick with convolutions replaced with AC Blocks, run
    ```
    python create_AC_cfqk.py
    ```
4. To convert the trained cifar-quick model into deploy mode, run
    ```
    python deploy_AC_cfqk.py
    ```
5. Use it in your own project
    
    Ac_Block_utils.py contains all functions for you to use it in your own projects. 
    
    First you'll need to use ```AC_Block()``` to build your model, it takes in almost the same parameters as the normal ```keras.layers.Conv2D()``` and ```keras.layers.BatchNormalization()```. Note you must pass in the name of each module for deploying purpose.

    After you've trained and saved your model with AC modules, use the ```deploy()``` function to convert it to deploy mode. Simply pass in the checkpoint file in .h5 format and a list of AC module names. It will convert it to deply mode and save a new model for you. Note that the converted model is not compiled yet, you'll need to compile it afterwards. 

    All functionalities only support tf.keras API for now.
    
