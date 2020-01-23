# 🌊 Deep Sort Tracker
Implementation of Deep Sort Tracking with Tensorflow object detection

## 🎈 Available Codes
1. [`auto_track`](track_me.py)
 
## ⚙ 🔩 Usage
0. Clone this repo
1. Open [DeepSortTracker](../DeepSortTracker) folder in CMD
2. Run:
   
    `python auto_track.py --video_file \path\to\your\video.mp4 --model_path \path\to\your\tf_model.pb --tracker \path\to\your\tracking_tf_model.pb` 

3. To see running options write:
   
   `python track_me.py --help`

4. Objects will be detected and tracked 🤗
5. Press <kbd>Q</kbd> to exit  

## 👀 Example
- 🕵️‍♀️ Detection model: [SSDLİte + MobileNet](http://download.tensorflow.org/models/object_detection/ssdlite_mobilenet_v2_coco_2018_05_09.tar.gz)
  - 🧐 See more models at [Tensorflow Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) 
- 👮‍♀️ Threshold: 0.6
  
![](../../res/deep_sort_output.gif)

> 🙄 Better results can be obtained when more powerful detection model is used

## 🤹‍♀️ More Arguments
- `--threshold`
- `--output_path`

## 👩‍🏫 Deep Sort Tracking
[🚀 GitHub Repository](https://github.com/nwojke/deep_sort)

## 💼 For Contact or Support
Find me on [LinkedIn](https://www.linkedin.com/in/asmaamirkhan/) and feel free to mail me, [Asmaa](mailto:asmaamirkhan.am@gmail.com) 🦋
