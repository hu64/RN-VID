# from: https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_lucas_kanade.html
import cv2
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import os


def get_pairs_from_list(list_names=['/store/datasets/KITTI/3tpfl/train.csv',
                                    '/store/datasets/KITTI/3tpfl/val.csv',
                                    '/store/datasets/KITTI/3tpfl/test.csv']):

    images = set()
    for list_name in list_names:
        with open(list_name, 'rb') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                img1_path = row[0]
                img0_path = img1_path[:-6] + '02' + '.png'
                images.add((img0_path, img1_path))
    return images


def normalize_2d_between_range(vector, min_range=0, max_range=255):

    max_vec = np.max(vector)
    min_vec = np.min(vector)
    vector = ((((vector - min_vec) / (max_vec - min_vec) - 0.5) * max_range) + (max_range / 2.0)) + min_range
    return vector


def dense_optical_flow(path0, path1):

    img0 = cv2.imread(path0)
    img1 = cv2.imread(path1)

    hsv = np.zeros_like(img0)
    hsv[..., 1] = 255

    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(img0, img1, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # add all the flow in both directions, we don't care about direction, just norm here
    # flow = abs(flow[..., 0]) + abs(flow[..., 1])
    # normalize the flow to get an image
    # flow = np.round(normalize_2d_between_range(flow)).astype(np.uint8)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return flow


def binarize_flow(img):

    # img = cv2.GaussianBlur(img, (5, 5), 0)
    # img = cv2.fastNlMeansDenoising(img)

    # Otsu's thresholding
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)

    img = (img > (ret/2)) * (np.zeros(img.shape)+255)

    # img -= int(ret)
    # np.clip(img, 0, 255)
    # img *= 1000
    # np.clip(img, 0, 255)

    return img


pairs = get_pairs_from_list()
pairs = sorted(pairs)
total = len(pairs)
current = 0
for pair in pairs:
    current += 1
    if current % 100 == 0:
        print(str(current) + ' / ' + str(total))

    flow_name = os.path.join(os.path.dirname(pair[1]), 'flow_b', os.path.split(pair[1])[1])

    flow = dense_optical_flow(pair[0], pair[1])
    # flow = binarize_flow(flow)

    # print(flow_name)
    if not os.path.exists(os.path.dirname(flow_name)):
        os.mkdir(os.path.dirname(flow_name))

    cv2.imwrite(flow_name, flow)


