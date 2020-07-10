# Football-players-detection
Various tools, scripts and research made for football players detection and extraction from one static video

## Description
Project is for master thesis research purposes but I thought some of functions may be usefull so made it public.

### Purpose
The main purpose of this project is to research various techniques and ways of detecting players. Ideally program would recognize players, team they belong to and track them after some initialisation.


## Getting Started
At this moment most of functions are adapted to particural video. You need to make some changes in order to work it for you (Check Usage section)
### Prerequisites
If you want to work with same video as me download it from here (393MB): [Google Drive](https://drive.google.com/file/d/1AfZjTKG3le_1MTOvFHcOUIhk51u1JraU/view?usp=sharing)

All Code was tested on Python 3.6.6

Install OpenCV (min. 4.1.1)
```
pip3 install opencv-python
```
Install PyTorch (min 1.3.1)
```
pip3 install torch===1.3.1 torchvision===0.4.2 -f https://download.pytorch.org/whl/torch_stable.html
```
Install imutils
```
pip3 install imutils
```

## Usage
### Quick usage
Download video from link above and use
```
python3 main.py
```
### Quick overview of scripts
###### main.py
All scripts and functions gathered up together. 
###### vision.py
Various functions connected with computer vision.
###### player.py
Class representing football player.
###### neural.py
Load neural network model for translating video points to 2D field representation.
###### learnpoints.py
Learn neural network model for translating sample video points to 2D field (*field.png*).
###### kalman.py
Prepare kalman filter class for player movement predictions
###### homography.py
Homography matrix for point from video to 2D translation

### Wiki
https://github.com/Th3NiKo/Football-players-detection/wiki
