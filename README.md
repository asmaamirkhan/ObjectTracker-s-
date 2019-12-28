# 🛰 Object Trackers
Implementation of object trackers that are provided in OpenCV library

## 🎈 Available Codes
0. [Single Object Tracker](/src/track_me.py)
0. [Multiple Object Tracker](/src/track_us.py)
 
> Make sure that you have OpenCV already installed

## 🌞 Available Trackers
* CSRT
* KCF
* BOOSTING
* MIL
* TLD
* MEDIANFLOW
* MOSSE

## 🚧 Attention 
These algorithms aren't powerful enough, for better tracking algorithms look at [SORT](https://github.com/abewley/sort) algorithm and [DEEP-SORT ✨](https://github.com/nwojke/deep_sort) algorithm.

## ⚙ 🔩 Usage
0. Clone this repo
0. Open [src](/src) folder in CMD
0. Write:
   
    `python track_me.py --video_file C:\your\path\to\your\video.mp4`

0. To see running options write:
   
   `python track_me.py --help`

0. The video will pop up, press `s` select your ROI (Region of Interest) that will be tracked
0. Press <kbd>Enter</kbd>
0. See your object while it is being tracked 🤗
0. Press <kbd>Q</kbd> to exit  

> Repeat **4** for each object in multiple object tracking

## 👀 Examples

### 1️⃣ Single Object Tracking
![](./res/single_output.gif)


### 2️⃣ Multi Object Tracking
![](./res/multi_output.gif)


## 👩‍🏫 Tracking Explanation
[Object Tracking in OpenCV](https://ehsangazar.com/object-tracking-with-opencv-fd18ccdd7369)

## 💼 For Contact or Support
Find me on [LinkedIn](https://www.linkedin.com/in/asmaamirkhan/) and feel free to mail me, [Asmaa](mailto:asmaamirkhan.am@gmail.com) 🦋
