import argparse
import cv2 as cv

def choose_tracker(tracker):
    OPENCV_OBJECT_TRACKERS = {
    "csrt": cv.TrackerCSRT_create,
    "kcf": cv.TrackerKCF_create,
    "boosting": cv.TrackerBoosting_create,
    "mil": cv.TrackerMIL_create,
    "tld": cv.TrackerTLD_create,
    "medianflow": cv.TrackerMedianFlow_create,
    "mosse": cv.TrackerMOSSE_create
    }

    if tracker not in OPENCV_OBJECT_TRACKERS:
        wanted_tracker = OPENCV_OBJECT_TRACKERS['kcf']()
        print('Invalid tracker, kcf used instead')
    else:
        wanted_tracker = OPENCV_OBJECT_TRACKERS[tracker]()
    return wanted_tracker

def track(wanted_tracker, video_file, tracker):
    tracking_win = "Kalman Object Tracking"
    cropped_win = "Tracked Region"
    cap = cv.VideoCapture(video_file)
    frame_counter = 0
    init_box = None
    while True:
        r, frame = cap.read()

        frame = cv.resize(frame, (700, 700))
        frame_counter += 1

        if frame is None:
            break

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord("s"):
            init_box = cv.selectROI(tracking_win, frame, fromCenter=False,
                                    showCrosshair=True)
            wanted_tracker.init(frame, init_box)

        if init_box is not None:
            success, box = wanted_tracker.update(frame)

            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv.rectangle(frame, (x, y), (x + w, y + h),
                            (0, 255, 0), 2)
                crop = frame[y:y+h, x:x+w]
                cv.imshow(cropped_win, crop)
            else:
                cv.destroyWindow(cropped_win)
            cv.putText(frame, 'Tracker: {}'.format(tracker), (30, 30),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv.imshow(tracking_win, frame)

def main(args):
    wanted_tracker = choose_tracker(args.tracker)
    track(wanted_tracker, args.video_file, args.tracker)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object tracking parameters')
    parser.add_argument('-v', '--video_file',
                        help='Path to your video file', type=str)
    parser.add_argument('-t', '--tracker',
                        help='Object tracking algorithm', type=str, default='kcf')
    args = parser.parse_args()
    main(args)