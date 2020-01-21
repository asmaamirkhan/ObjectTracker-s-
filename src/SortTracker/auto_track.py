from sort.sort import *
from DetectorAPI import DetectorAPI
import cv2 as cv
import argparse
import os

def main(args):
    dapi = DetectorAPI(args.model_path)
    cap = cv.VideoCapture(args.video_file)
    tracker = Sort()
    
    while True:
        r, frame = cap.read()
        cv.imshow("Output", frame)
        key = cv.waitKey(1)
        if key & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Object tracking parameters')
    parser.add_argument('-v', '--video_file',
                        help='Path to your video file', type=str, required=True)
    parser.add_argument('-m', '--model_path',
                        help='Path to Tensorflow model (.pb)', type=str, required=True)
    parser.add_argument('-o', '--output_path',
                        help='Output file path', type=str)
    args = parser.parse_args()
    assert os.path.isfile(args.video_file), 'Invalid input file'
    assert os.path.isfile(args.model_path), 'Invalid model path'
    if args.output_path:
        assert os.path.isdir(os.path.dirname(
            args.output_path)), 'No such directory'

    main(args)
