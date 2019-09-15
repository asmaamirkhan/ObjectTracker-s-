# author: Asmaa Mirkhan ~ 2019


import argparse
import cv2 as cv
import time
import os.path


def init_tracker(tracker):
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv.TrackerCSRT_create,
        "kcf": cv.TrackerKCF_create,
        "boosting": cv.TrackerBoosting_create,
        "mil": cv.TrackerMIL_create,
        "tld": cv.TrackerTLD_create,
        "medianflow": cv.TrackerMedianFlow_create,
        "mosse": cv.TrackerMOSSE_create
    }

    wanted_tracker = OPENCV_OBJECT_TRACKERS[tracker]()

    return wanted_tracker


def track( args):
    tracking_win = "Object Tracking"
    cropped_win = "Tracked Region"
    cap = cv.VideoCapture(args.video_file)
    if args.output_path:
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        output = cv.VideoWriter(
            args.output_path + '.mp4', fourcc, 20.0, (700, 700))
    frame_counter = 0
    init_box = None
    trackers = cv.MultiTracker_create()
    while True:
        r, frame = cap.read()

        frame_counter += 1
        start = time.clock()
        frame = cv.resize(frame, (700, 700))

        if frame is None:
            break

        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

        if key & 0xFF == ord("s"):
            init_box = cv.selectROI(tracking_win, frame, fromCenter=False,
                                    showCrosshair=True)
            tracker = init_tracker(args.tracker)
            trackers.add(tracker, frame, init_box)

        if init_box is not None:
            success, boxes = trackers.update(frame)

            if success:
                for box in boxes:
                    (x, y, w, h) = [int(v) for v in box]
                    cv.rectangle(frame, (x, y), (x + w, y + h),
                                 (0, 255, 0), 2)

        cv.putText(frame, 'Tracker: {}, Frame: {}'.format(args.tracker, frame_counter), (30, 30),
                   cv.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 1)
        cv.imshow(tracking_win, frame)
        end = time.clock()
        print('Frame: {} , Elapsed time: {:.3f}'.format(frame_counter, end-start))
        print('=========================================')
        if args.output_path:
            output.write(frame)



if __name__ == "__main__":
    tracking_methods = ["csrt", "kcf", "boosting", "mil",
                        "tld", "medianflow", "mosse"]
    parser = argparse.ArgumentParser(description='Object tracking parameters')
    parser.add_argument('-v', '--video_file',
                        help='Path to your video file', type=str, required=True)
    parser.add_argument('-t', '--tracker',
                        help='Object tracking algorithm', type=str, default='kcf')
    parser.add_argument('-o', '--output_path',
                        help='Output file path', type=str)
    args = parser.parse_args()

    if args.tracker not in tracking_methods:
        print('Invalid tracker name, kcf used instead')
        args.tracker = 'kcf'

    assert os.path.isfile(args.video_file), 'Invalid input file'
    if args.output_path:
        assert os.path.isdir(os.path.dirname(
            args.output_path)), 'No such directory'

    track(args)
