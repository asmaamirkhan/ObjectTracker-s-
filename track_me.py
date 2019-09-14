import argparse 
import cv2 as cv




parser = argparse.ArgumentParser(description='Object tracking parameters')
parser.add_argument('-v','--video_file', help='Path to your video file', type=str)
parser.add_argument('-t','--tracker', help='Object tracking algorithm', type=str, default='kcf')
args = parser.parse_args()


OPENCV_OBJECT_TRACKERS = {
		"csrt": cv.TrackerCSRT_create,
		"kcf": cv.TrackerKCF_create,
		"boosting": cv.TrackerBoosting_create,
		"mil": cv.TrackerMIL_create,
		"tld": cv.TrackerTLD_create,
		"medianflow": cv.TrackerMedianFlow_create,
		"mosse": cv.TrackerMOSSE_create
	}

if args.tracker not in OPENCV_OBJECT_TRACKERS:
    wanted_tracker = OPENCV_OBJECT_TRACKERS['kcf']()
    print('Invalid tracker, kcf used instead')
else:
    wanted_tracker=OPENCV_OBJECT_TRACKERS[args.tracker]()

print(wanted_tracker)

tracking_win = "Kalman Object Tracking"
filter_win = "Hue histogram back projection"
cropped_win = "initial selected region"


cap = cv.VideoCapture(args.video_file)

while True:
    r, frame = cap.read()
    cv.imshow(tracking_win, frame)
    key = cv.waitKey(1)
    if key & 0xFF == ord('q'):
        break