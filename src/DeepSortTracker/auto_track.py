import os
from DetectorAPI import DetectorAPI
import cv2 as cv
import argparse
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection as ddet
from deep_sort.tools import generate_detections as gdet


WINDOW_TITLE = 'Tracker'


def main(args):
    detector = DetectorAPI(args.model_path)
    cap = cv.VideoCapture(args.video_file)
    while True:
        check, frame = cap.read()
        sort_tracking_params = []

        boxes, scores, classes, number = detector.processFrame(frame,
                                                               debug_time=True)

        for index, box in enumerate(boxes):
            if scores[index] > args.threshold:
                cv.rectangle(frame, (box[1], box[0]), (box[3], box[2]),
                             (255, 255, 255),
                             thickness=2)

        cv.imshow(WINDOW_TITLE, frame)

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
                        help='Path to Tensorflow object detection model (.pb)',
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
    parser.add_argument('-r',
                        '--tracker',
                        help='Path to deep sort Tensorflow model .pb',
                        type=str,
                        required=True)
    args = parser.parse_args()

    assert os.path.isfile(args.video_file), 'Invalid input file'
    assert os.path.isfile(args.model_path), 'Invalid model path'
    assert os.path.isfile(args.tracker), 'Invalid deep sort model file'
    if args.output_path:
        assert os.path.isdir(os.path.dirname(
            args.output_path)), 'No such directory'

    main(args)
