import argparse 
import cv2 as cv
from imutils.video import FPS



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
fps = None
initBB = None
while True:
    r, frame = cap.read()
    if frame is None:
        break
    
    key = cv.waitKey(1)
    if key & 0xFF == ord('q'):
        break

    if key & 0xFF == ord("s"):
        initBB = cv.selectROI(tracking_win, frame, fromCenter=False,
			showCrosshair=True)
        wanted_tracker.init(frame, initBB)
        fps = FPS().start()
    if initBB is not None:
        success, box = wanted_tracker.update(frame)
        print(box)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            print(x,y,w,h)
            cv.rectangle(frame, (x, y), (x + w, y + h),
				(0, 255, 0), 2)
 
		# update the FPS counter
        fps.update()
        fps.stop()
 
		# initialize the set of information we'll be displaying on
		# the frame
        info = [
			("Tracker", wanted_tracker),
			("Success", "Yes" if success else "No"),
			("FPS", "{:.2f}".format(fps.fps())),
		]
    

		# loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv.putText(frame, text, (10, 10),
			    cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv.imshow(tracking_win, frame)

                