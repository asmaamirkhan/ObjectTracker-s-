import os
import cv2 as cv
import argparse
from detection_utils.DetectorAPI import DetectorAPI
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection as ddet
from deep_sort.tools import generate_detections as gdet

WINDOW_TITLE = 'Tracker'

MAX_COSINE_DISTANCE = 0.3
NN_BUDGET = None
NMS_MAX_OVERLAP = 1.0
METRIC = nn_matching.NearestNeighborDistanceMetric("cosine",
                                                   MAX_COSINE_DISTANCE,
                                                   NN_BUDGET)


def adapt_to_deep_sort(boxes):
    for box in boxes:
        box = list(box)
        box[0], box[1] = box[1], box[0]
        box[2], box[3] = box[3], box[2]
    return boxes


def main(args):
    # setting deep sort parameters
    tracker = Tracker(METRIC)
    encoder = gdet.create_box_encoder(args.tracker, batch_size=1)

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

    # read the video frame by frame
    while True:
        check, frame = cap.read()

        # initialize tracking parameters
        sort_tracking_params = []

        # do real detection
        # boxes are in (x_top_left, y_top_left, x_bottom_right, y_bottom_right) format
        boxes, scores, classes, number = detector.processFrame(frame,
                                                               debug_time=True)

        # filter boxes due to threshold
        boxes = [
            boxes[i] for i in range(0, number) if scores[i] > args.threshold
        ]

        # do tracking
        boxes = adapt_to_deep_sort(boxes)
        features = encoder(frame, boxes)
        detections = [
            Detection(tbox, 1.0, feature)
            for tbox, feature in zip(boxes, features)
        ]
        tracker.predict()
        tracker.update(detections)

        # draw boxes due to tracking results
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            tbox = track.to_tlbr()
            cv.rectangle(frame, (int(tbox[1]), int(tbox[0])),
                         (int(tbox[3]), int(tbox[2])), (255, 255, 255),
                         thickness=2)
            cv.putText(frame,
                       'id: {}'.format(track.track_id),
                       (int(tbox[1]), int(tbox[0])),
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
