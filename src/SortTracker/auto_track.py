import os
import argparse
import cv2 as cv
from sort.sort import *
from detection_utils.DetectorAPI import DetectorAPI
from detection_utils.ms_coco_classnames import MS_CLASSES

WINDOW_TITLE = 'Tracker'


def clean_result(result, number):
    return result[:number]


def main(args):
    # Initialize the object detector
    detector = DetectorAPI(args.model_path)
    # open the video
    cap = cv.VideoCapture(args.video_file)

    # create output file
    if args.output_path:
        fourcc = cv.VideoWriter_fourcc(*'MP4V')
        # cap.get(3) = width, cap.get(4) = height
        output = cv.VideoWriter(args.output_path, fourcc, 20.0,
                                (int(cap.get(3)), int(cap.get(4))))

    # initialize tracker
    tracker = Sort()

    # read frame by frame
    while True:
        check, frame = cap.read()
        sort_tracking_params = []

        # do real detection
        # boxes are in (x_top_left, y_top_left, x_bottom_right, y_bottom_right) format
        boxes, scores, classes, number = detector.processFrame(frame,
                                                               debug_time=True)

        # keep only valid results
        boxes = clean_result(boxes, number)
        scores = clean_result(scores, number)
        classes = clean_result(classes, number)

        # prepare tracking parameters, list of [x1,y1,x2,y2,score]
        for i, box in enumerate(boxes):
            sort_tracking_params.append(
                [box[0], box[1], box[2], box[3], scores[i]])

        sort_tracking_params = np.array(sort_tracking_params)
        trackers = tracker.update(sort_tracking_params).astype(np.int32)

        # draw boxes due to tracking results
        for index, box in enumerate(trackers):
            if scores[index] > args.threshold:
                cv.rectangle(frame, (box[1], box[0]), (box[3], box[2]),
                             (255, 255, 255),
                             thickness=2)
                cv.putText(frame,
                           'class: {}, id: {}'.format(
                               MS_CLASSES[classes[index]], box[4]),
                           (box[1], box[0]),
                           cv.FONT_HERSHEY_PLAIN,
                           1, (255, 255, 255),
                           thickness=2)

        cv.imshow(WINDOW_TITLE, frame)

        # save frames to specified path
        if args.output_path:
            output.write(frame)

        # the end of the video?
        if not check:
            break
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object tracking parameters')
    parser.add_argument('-v',
                        '--video_file',
                        help='Path to your video file',
                        type=str,
                        required=True)
    parser.add_argument('-m',
                        '--model_path',
                        help='Path to Tensorflow model (.pb)',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output_path',
                        help='Output file path',
                        type=str)
    parser.add_argument('-t',
                        '--threshold',
                        help="Object detection threshold, default = 0.7",
                        type=float,
                        default=0.7)
    args = parser.parse_args()

    assert os.path.isfile(args.video_file), 'Invalid input file'
    assert os.path.isfile(args.model_path), 'Invalid model path'
    if args.output_path:
        assert os.path.isdir(os.path.dirname(
            args.output_path)), 'No such directory'

    main(args)
