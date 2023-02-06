# Circle Detection with CNN

Author: Jiarui Li <br/>
Date: Feb 5, 2023

-----
Table of Content

[Overview](#overview)<br/>
[Code Structure](#struct)<br/>
[Development Process](#dev)<br/>
[Training & Evaluation](#train_eval)<br/>
[Futhre Improvements](#improve)<br/>
[Simple Launch](#pip)<br/>

------

## Overview<a id="overview"></a>
This project implement a deep CNN to detect the location of a circle given a noisy image. The images are 100x100 (by default) and one channel, generated with the given starter codes. The image demo for cirlce [with](https://github.com/U1ltra/circleDetection/tree/main/experiments/noise50/img/sample_noisy.png) or [without](https://github.com/U1ltra/circleDetection/tree/main/experiments/noise50/img/sample.png) noise. The task of this project is using the noise image to recover the circle's center (x,y) and its radius.
<!-- my work -->
My contribution is mainly composed of the following parts,
- Built a compact python package for this circle detection task. Details are explained below.
- Tested the model based on some metrics and provided some insights
<!-- final results -->
Eventually, my best model reached **91.93% average IOU** on test dataset with 1000 data points, generated with 0.5 noise level. The **average test loss** on each point is **19.06**. And percentage of **IOU > 0.8** and **IOU > 0.9** are **93%** and **68.7%**, respectively.


## Code Structure<a id="struct"></a>
This package is organized into the following structure.

    circleDetection
        |- experiments
        |- notebook
        |- circleGenerator.py
        |- dataset.py
        |- eval.py
        |- model.py
        |- README.py
        |- trainer.py
        |- util.py

The *experiments* directory contains two selected trained models, original test data, and stored training as well as validaiton loss. This is intended for the user to reproduce the experiment results very quickly.

The *circleGenerator.py* module contains most of the starter code. This is mainly used to define circles and support the generation of circle image dataset.

The *dataset.py* module defines the dataset wrapper and main dataset generator function.

The *eval.py* module contains the logic for model testing after training has be finished. The metrics are evaluated through this script.

The *model.py* module defines the architecture of the CNN network used in this project.

The *trainer.py* is the module for launching the training.

The *util.py* defines a lot of useful helper functions used throughout other modules.

## Development Process<a id="dev"></a>
To get started, three datasets, train, valid, and test, are generated. In order to get a sense of how much data might be needed. I referred to the popular ImageNet dataset and comed up with the datasize setting in *trainer.py*.

With dataset settled, the next step is data preprocessings. First, the data generated are converted into numpy ndarray features and numpy ndarray labels. Second, in order to fit into torch's data loader, I define a *Dataset* wrapper that will store the feature and label arrays and return them in pairs. One more thing to notice is that generally images have three channels. For the generated circle images, however, it only have one channel, since it is a 2-D numpy adarry. To apply CNN kernel on this 2-D image, I need to expend each image by one dimension to indicate that the image only have one channel.

When dataset is prepared, the network architecture is then set (see detail in *model.py*). I used a CNN with about 7M parameters.

Then, the training can be launched. With the following triaining setting, it takes about one hour to run on a machine with a single GPU.
```python
epoch_num = 200
batch_size = 128

lr = 1e-3
momentum = 0.9
weight_decay = 1e-5

criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

train_datasize = 3*1e4
valid_datasize = 1e3
test_datasize = 1e3
```

## Training & Evaluation<a id="train_eval"></a>
With the best model (best validation loss) obtained from the training, I calculated the following evaluation results. I mainly inspect three metrics, Average IOU, rate of IOU > 0.8, and rate of IOU > 0.9. I not only evaluated the model on the test data generated together with training and validation data. I further test the model's generalization ability by testing the model on new test data with different random seed and noise level. The following three tables are the results.

According to the result, the model generalizes better towards lower noise level data, while performs worse on higher noise level data. It is intuitively true that the model should do worse on more noisy data. However, one might expect better performance on less noisy data. I think the performance drop might come from the model's resistance to noise. Since in images with higher noist, the model would recognize a lot of signals to be noise. When inferencing on less noisy data, the model still try to be resistent to noise, while this time there are more signals than noise being omitted.

|         Metrics          |    Noise Level   |    Result   |
| :-------------------------: |:----------------:| --------------:|
|Avg IOU|0.7|38.51%|
|Avg IOU|0.6|78.40%|
|Avg IOU|0.5 - original seed|97.26%|
|Avg IOU|0.5 - new seed|96.66%|
|Avg IOU|0.4|96.14%|
|Avg IOU|0.3|90.30%|

|         Metrics          |    Noise Level   |    Result   |
| :-------------------------: |:----------------:| --------------:|
|IOU > 0.8|0.7|20.30%|
|IOU > 0.8|0.6|68.10%|
|IOU > 0.8|0.5 - original seed|98.30%|
|IOU > 0.8|0.5 - new seed|97.50%|
|IOU > 0.8|0.4|97.40%|
|IOU > 0.8|0.3|86.20%|

|         Metrics          |    Noise Level   |    Result   |
| :-------------------------: |:----------------:| --------------:|
|IOU > 0.9|0.7|16.80%|
|IOU > 0.9|0.6|54.00%|
|IOU > 0.9|0.5 - original seed|93.50%|
|IOU > 0.9|0.5 - new seed|92.60%|
|IOU > 0.9|0.4|91.70%|
|IOU > 0.9|0.3|75.70%|

## Futhre Improvements<a id="improve"></a>
This package can be further improved in several ways.
1. Data Preprocessing <br/>
I tried to use some signal processing techniques, such as gaussian filtering, mean filtering, median filtering, and sobel, to preprocess the images. However, the result does not seem to be useful at first glance. Thus, they are not incooporated into the current implementation. In low or median noise setting, as it turns out, CNN itself is strong enough to extract the information from the dataset. In the higher noise level, using some signal processing strategies to augment the data might be necessary.

2. Axis Mapping <br/>
Another possible way to make the data more learner friendly is to map the x, y axises to polar axises, which will map the cirlce to a straight line.

3. Architecture and Hyperparameter Tuning <br/>
This is another aspect that could bring significant performance enhancement. In this project, I have not tune the architecture and hyperparameters too much. It is possible to use more fansy architectures, such as ResNet, VGG, and better hyperparameter set to further improve the preformance.


## Simple Launch<a id="pip"></a>
The following is a simple way to launch this package.
```
# create conda env
conda create -n ML_challenge python==3.8.10
conda activate ML_challenge

# install dependency
pip3 install -r requirements.txt

# launch training
python3 trainer.py

# perform evaluation
python3 eval.py
```
<!-- Run the notebook with docker -->

<!-- ## Problems Encountered -->
<!--  -->
