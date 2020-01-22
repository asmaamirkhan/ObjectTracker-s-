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

MAX_COSINE_DISTANCE = 0.3
NN_BUDGET = None
NMS_MAX_OVERLAP = 1.0
METRIC = nn_matching.NearestNeighborDistanceMetric("cosine",
                                                   MAX_COSINE_DISTANCE,
                                                   NN_BUDGET)


def main(args):
    tracker = Tracker(METRIC)
    encoder = gdet.create_box_encoder(args.tracker, batch_size=1)
    detector = DetectorAPI(args.model_path)
    cap = cv.VideoCapture(args.video_file)
    while True:
        check, frame = cap.read()
        sort_tracking_params = []

        boxes, scores, classes, number = detector.processFrame(frame,
                                                               debug_time=True)
        boxes = [boxes[i] for i in range(0, number) if scores[i] > args.threshold]

        for box in boxes:
            box = list(box)
            box[0], box[1] = box[1], box[0]
            box[2], box[3] = box[3], box[2]

        features = encoder(frame, boxes)
        detections = [
            Detection(bbox, 1.0, feature)
            for bbox, feature in zip(boxes, features)
        ]
        tracker.predict()
        tracker.update(detections)

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            cv.rectangle(frame, (int(bbox[1]), int(bbox[0])),
                         (int(bbox[3]), int(bbox[2])), (255, 255, 255),
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
