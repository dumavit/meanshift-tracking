import os
import enum
import argparse
import cv2
import numpy as np

from config import dataset_config

GREEN_COLOR = (0, 255, 0)
RECTANGLE_THICKNESS = 2

INPUT_FOLDER = 'input'
IMAGES_FOLDER = 'img'
OUTPUT_FOLDER = 'output'

# Setup the termination criteria, either 1000 iterations or move by at least 1 pt
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1)

HISTOGRAM_SIZE = [180]
# Disable scaling
SCALE = 1
# Histogram bin boundaries
RANGES = [0, 180]
VIDEO_FPS = 30


class Methods(enum.Enum):
    MeanShift = 'meanshift'
    CamShift = 'camshift'


def get_images(dataset):
    # Sort images in ascending order
    dataset_path = os.path.join(INPUT_FOLDER, dataset, IMAGES_FOLDER)
    names = sorted(os.listdir(dataset_path), key=lambda name: int(name.split('.')[0]))
    paths = (os.path.join(dataset_path, name) for name in names)
    return [cv2.imread(path) for path in paths]


def get_roi_hist(image, args):
    x, y, width, height = args.roi
    roi_image = image[y:y + height, x:x + width]
    hsv_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HSV)

    channel = dataset_config[args.dataset]['channel']
    filter_from = dataset_config[args.dataset]['filter_from']
    filter_to = dataset_config[args.dataset]['filter_to']
    mask = cv2.inRange(hsv_roi, np.array(filter_from), np.array(filter_to))

    roi_hist = cv2.calcHist([hsv_roi], channel, mask, HISTOGRAM_SIZE, RANGES)
    roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    return roi_hist


def write_video(images, dataset, method):
    print('Generating video')
    # Save video as .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height, width, layers = images[0].shape
    video_path = os.path.join(OUTPUT_FOLDER, method, dataset + '.mp4')
    video = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (width, height))

    for image in images:
        video.write(image)

    cv2.destroyAllWindows()
    video.release()
    print('Done')


def meanshift(images, args):
    x, y, width, height = args.roi

    first_frame, *images = images

    channel = dataset_config[args.dataset]['channel']
    roi_hist = get_roi_hist(first_frame, args)

    print('Start tracking')
    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.calcBackProject([hsv], channel, roi_hist, RANGES, SCALE)
        _, track_window = cv2.meanShift(mask, (x, y, width, height), TERM_CRITERIA)
        x, y, w, h = track_window
        cv2.rectangle(image, (x, y), (x + w, y + h), GREEN_COLOR, RECTANGLE_THICKNESS)

        cv2.imshow('Mask', mask)
        cv2.imshow('Frame', image)
        # Show each frame for 40 ms
        key = cv2.waitKey(40)
        # Break on esc key
        if key == 27:
            break

    cv2.destroyAllWindows()
    write_video(images, args.dataset, 'meanshift')


def camshist(images, args):
    x, y, width, height = args.roi
    first_frame, *images = images

    channel = dataset_config[args.dataset]['channel']
    roi_hist = get_roi_hist(first_frame, args)

    print('Start tracking')
    for image in images:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        mask = cv2.calcBackProject([hsv], channel, roi_hist, RANGES, SCALE)

        ret, track_window = cv2.CamShift(mask, (x, y, width, height), TERM_CRITERIA)
        pts = cv2.boxPoints(ret)
        pts = np.int0(pts)
        cv2.polylines(image, [pts], True, (0, 255, 0), 2)
        cv2.imshow('Mask', mask)
        cv2.imshow('Frame', image)

        # Show each frame for 40 ms
        key = cv2.waitKey(40)
        # Break on esc key
        if key == 27:
            break

    cv2.destroyAllWindows()
    write_video(images, args.dataset, 'camshift')


def main(args):
    if args.dataset not in dataset_config:
        print('Dataset not found')
        return

    if not args.roi:
        # Use default roi from config
        args.roi = dataset_config[args.dataset]['roi']

    images = get_images(args.dataset)

    method = Methods(args.method)

    if method == Methods.MeanShift:
        meanshift(images, args)
    elif method == Methods.CamShift:
        camshist(images, args)
    else:
        print('Method not found')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MeanShift and CamShift object detectors')
    parser.add_argument('--dataset', type=str, help='Dataset name', required=True)
    parser.add_argument('--roi', type=int, nargs=4, help='Region of interest (x, y, width, height)')
    parser.add_argument('--method', type=str, help='meanshift or camshift method', default=Methods.MeanShift)

    args = parser.parse_args()
    main(args)
