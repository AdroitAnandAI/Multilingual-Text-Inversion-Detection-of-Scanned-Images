# Multilingual Text Inversion Detection of Scanned Images
## _Text Localization, Image Inversion Detection of Scanned Documents & Language Identification based on Shape Context and CV_

_Research Paper: Multilingual Text Inversion Detection using Shape Context, presented in the IEEE TENSYMP 2021 Conference held at Grand Hyatt Jeju, the Republic of Korea on 23-25th Aug 2021._

_Paper Link:_
https://ieeexplore.ieee.org/document/9550858

_Paper Presentation_
https://youtu.be/zm9uaxdWMOA


## Problem Definition
There can be problems in textual scanned images. The problem of inversion is one of the hardest anomaly to detect efficiently though it can be easily decipherable visually. Moreover, the scanned document can be in any language and the text can be anywhere in the image. 

In this project, an algorithm to efficiently localize text has been implemented. Once the text area in an image is localized, it is passed on to language identification algorithm. Further, a mathematical descriptor is used to identify the text is inverted or not. The entire pipeline uses traditional methods in place of deep learning based methods and hence much more efficient.


## How to run:
```
docker pull karthik199712/computer_vision:cv
```
To execute the pipeline with default image:
```
sudo docker run -it karthik199712/computer_vision:cv main.py
```
To execute the pipeline with an image in the dataset, give the image path and name after --image flag. 

For instance, to execute with 11.png input image, command is as below:
```
sudo docker run -it karthik199712/computer_vision:cv main.py --image ./data/11.png
```
The deep learning implementation for comparison is available in VGG16_INFERENCE_BASE.ipynb.

You can find some output examples as below.

## Upright English Image
```
sudo docker run -it karthik199712/computer_vision:cv main.py --image ./data/0.png
```
<p align="center">
  <img src="upright.png">
</p> 

## Inverted English Image
```
sudo docker run -it karthik199712/computer_vision:cv main.py --image ./data/1.png
```
<p align="center">
  <img src="inverted.png">
</p> 

## Upright Malayalam Image
```
sudo docker run -it karthik199712/computer_vision:cv main.py --image ./data/mal1.png
```
<p align="center">
  <img src="mal_upright.png">
</p> 

## Inverted Malayalam Image
```
sudo docker run -it karthik199712/computer_vision:cv main.py --image ./data/mal2.png
```
<p align="center">
  <img src="mal_inverted.png">
</p> 

## Upright Greek Image
```
sudo docker run -it karthik199712/computer_vision:cv main.py --image ./data/greek1.png
```
<p align="center">
  <img src="greek_upright.png">
</p> 

## Inverted Greek Image
```
sudo docker run -it karthik199712/computer_vision:cv main.py --image ./data/greek2.png
```
<p align="center">
  <img src="greek_inverted.png">
</p> 

- Co-working Credits: [Karthik K](https://github.com/karthik1997)
