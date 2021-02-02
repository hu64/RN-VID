"""
Copyright 2017-2018 yhenon (https://github.com/yhenon/)
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from .generator import Generator
from ..utils.image import read_image_bgr

import numpy as np
from PIL import Image
from six import raise_from

import csv
import sys
import os.path
import cv2

def _parse(value, function, fmt):
    """
    Parse a string into a value, and format a nice ValueError if it fails.

    Returns `function(value)`.
    Any `ValueError` raised is catched and a new `ValueError` is raised
    with message `fmt.format(e)`, where `e` is the caught `ValueError`.
    """
    try:
        return function(value)
    except ValueError as e:
        raise_from(ValueError(fmt.format(e)), None)


def _read_classes(csv_reader):
    """ Parse the classes file given by csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise_from(ValueError('line {}: format should be \'class_name,class_id\''.format(line)), None)
        class_id = _parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


def _read_annotations(csv_reader, classes):
    """ Read annotations from the csv_reader.
    """
    result = {}
    for line, row in enumerate(csv_reader):
        line += 1

        try:
            img_file, x1, y1, x2, y2, class_name = row[:6]
        except ValueError:
            raise_from(ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)), None)

        if img_file not in result:
            result[img_file] = []

        # If a row contains only an image path, it's an image without annotations.
        if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
            continue

        x1 = _parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
        y1 = _parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
        x2 = _parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
        y2 = _parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

        # Check that the bounding box is valid.
        if x2 <= x1:
            raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
        if y2 <= y1:
            raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

        # check if the current class name is correctly present
        if class_name not in classes:
            raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

        result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
    return result


def _open_for_csv(path):
    """ Open a file with flags suitable for csv.reader.

    This is different for python2 it means with mode 'rb',
    for python3 this means 'r' with "universal newlines".
    """
    if sys.version_info[0] < 3:
        return open(path, 'rb')
    else:
        return open(path, 'r', newline='')


class CSVGenerator(Generator):
    """ Generate data for a custom CSV dataset.

    See https://github.com/fizyr/keras-retinanet#csv-datasets for more information.
    """

    def __init__(
        self,
        csv_data_file,
        csv_class_file,
        flow_flag='',
        base_dir=None,
        **kwargs
    ):
        """ Initialize a CSV data generator.

        Args
            csv_data_file: Path to the CSV annotations file.
            csv_class_file: Path to the CSV classes file.
            base_dir: Directory w.r.t. where the files are to be searched (defaults to the directory containing the csv_data_file).
        """
        self.image_names = []
        self.image_data  = {}
        self.base_dir    = base_dir
        self.flow_flag = flow_flag

        # Take base_dir from annotations file if not explicitly specified.
        if self.base_dir is None:
            self.base_dir = os.path.dirname(csv_data_file)

        # parse the provided class file
        try:
            with _open_for_csv(csv_class_file) as file:
                self.classes = _read_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise_from(ValueError('invalid CSV class file: {}: {}'.format(csv_class_file, e)), None)

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with _open_for_csv(csv_data_file) as file:
                self.image_data = _read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise_from(ValueError('invalid CSV annotations file: {}: {}'.format(csv_data_file, e)), None)
        self.image_names = list(self.image_data.keys())

        super(CSVGenerator, self).__init__(**kwargs)

    def size(self):
        """ Size of the dataset.
        """
        return len(self.image_names)

    def num_classes(self):
        """ Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def name_to_label(self, name):
        """ Map name to label.
        """
        return self.classes[name]

    def label_to_name(self, label):
        """ Map label to name.
        """
        return self.labels[label]

    def image_path(self, image_index):
        """ Returns the image path for image_index.
        """
        return os.path.join(self.base_dir, self.image_names[image_index])

    def image_aspect_ratio(self, image_index):
        """ Compute the aspect ratio for an image with image_index.
        """
        # PIL is fast for metadata
        image = Image.open(self.image_path(image_index))
        return float(image.width) / float(image.height)

    def load_image(self, image_index):
        """ Load an image at the image_index.
        """
        if self.flow_flag != '' and self.flow_flag != 'none':
            # print('here inside flow stuff')
            img1_path = self.image_path(image_index)
            if 'kitti' in self.flow_flag:
                img0_path = img1_path[:-6] + '02' + '.png'
            else:
                index = os.path.basename(img1_path).replace('.jpg', '')\
                    .replace('img', '')\
                    .replace('blurred_', '')\
                    .replace('gamma0_', '')\
                    .replace('gamma1_', '')\
                    .replace('noisy_', '')\
                    .replace('flip_', '')\
                    .replace('.png', '')\
                    .replace('.JPEG', '')
                if 'imageNet' not in self.flow_flag:
                    rest = img1_path.replace(index + '.jpg', '').replace(os.path.dirname(img1_path), '')
                else:
                    rest = img1_path.replace(index + '.JPEG', '').replace(os.path.dirname(img1_path), '')
                length = len(index)
                modulo = '1'
                for i in range(length):
                    modulo += '0'
                if int(index) == 1:
                    index = 2

                if 'espacement2' in self.flow_flag:
                    previous2 = rest + str((int(index) - 4) % int(modulo)).zfill(length)
                    previous = rest + str((int(index) - 2) % int(modulo)).zfill(length)
                    next = rest + str((int(index) + 2) % int(modulo)).zfill(length)
                    next2 = rest + str((int(index) + 4) % int(modulo)).zfill(length)
                elif 'espacement3' in self.flow_flag:
                    previous2 = rest + str((int(index) - 6) % int(modulo)).zfill(length)
                    previous = rest + str((int(index) - 3) % int(modulo)).zfill(length)
                    next = rest + str((int(index) + 3) % int(modulo)).zfill(length)
                    next2 = rest + str((int(index) + 6) % int(modulo)).zfill(length)
                else:
                    previous2 = rest + str((int(index) - 2) % int(modulo)).zfill(length)
                    previous = rest + str((int(index) - 1) % int(modulo)).zfill(length)
                    next = rest + str((int(index) + 1) % int(modulo)).zfill(length)
                    next2 = rest + str((int(index) + 2) % int(modulo)).zfill(length)

                if self.flow_flag == 'external':
                    img0_path = os.path.join(os.path.dirname(img1_path), 'flow', os.path.split(img1_path)[1])
                elif self.flow_flag == 'external_denoise':
                    img0_path = os.path.join(os.path.dirname(img1_path), 'flow_denoise', os.path.split(img1_path)[1])
                elif 'imageNet' in self.flow_flag:
                    img_1_path = os.path.dirname(img1_path) + previous2 + '.JPEG'
                    img0_path = os.path.dirname(img1_path) + previous + '.JPEG'
                    img2_path = os.path.dirname(img1_path) + next + '.JPEG'
                    img3_path = os.path.dirname(img1_path) + next2 + '.JPEG'
                else:
                    img_1_path = os.path.dirname(img1_path) + previous2 + '.jpg'
                    img0_path = os.path.dirname(img1_path) + previous + '.jpg'
                    img2_path = os.path.dirname(img1_path) + next + '.jpg'
                    img3_path = os.path.dirname(img1_path) + next2 + '.jpg'

            img0 = read_image_bgr(img0_path)
            img1 = read_image_bgr(img1_path)

            if 'flow_s' in self.base_dir or 'flow-y' in self.base_dir:
                img = np.concatenate((img0, img1), axis=2)
            elif '3d' in self.flow_flag:
                img = np.stack((img0, img1), axis=0)
            elif 'flow_c' in self.base_dir:
                img = np.concatenate((img0, img1), axis=2)
            elif 'three' in self.flow_flag:
                img2 = read_image_bgr(img2_path)
                img = np.concatenate((img0, img1, img2), axis=2)
            elif 'five' in self.flow_flag and 'flow' not in self.flow_flag:
                img_1 = read_image_bgr(img_1_path)
                img2 = read_image_bgr(img2_path)
                img3 = read_image_bgr(img3_path)
                img = np.concatenate((img_1, img0, img1, img2, img3), axis=2)
            elif 'five_flow' in self.flow_flag:
                img2 = read_image_bgr(img2_path)

                flow1 = cv2.calcOpticalFlowFarneback(cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY),
                                                     cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                                                     None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag1, ang1 = cv2.cartToPolar(flow1[..., 0], flow1[..., 1])
                flow2 = cv2.calcOpticalFlowFarneback(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY),
                                                     cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY),
                                                     None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag2, ang2 = cv2.cartToPolar(flow2[..., 0], flow2[..., 1])

                mag1 = np.abs(mag1)
                mag2 = np.abs(mag2)

                # mag1 = ((mag1 / np.max(mag1)) * 255).astype(np.uint8)
                # mag2 = ((mag2 / np.max(mag2)) * 255).astype(np.uint8)

                # cv2.imwrite('/usagers2/huper/dev/downloads/samples/' + str(self.group_index) + '-1.jpg', mag1)
                # cv2.imwrite('/usagers2/huper/dev/downloads/samples/' + str(self.group_index) + '-2.jpg', mag2)

                mag1 = np.expand_dims(mag1, axis=2)
                mag2 = np.expand_dims(mag2, axis=2)

                img = np.concatenate((img0, img1, img2, mag1, mag2), axis=2)
            else:
                img = np.concatenate((img0, img1), axis=2)
            return img
        else:
            return read_image_bgr(self.image_path(image_index))

    def load_annotations(self, image_index):
        """ Load annotations for an image_index.
        """
        path   = self.image_names[image_index]
        annots = self.image_data[path]
        boxes  = np.zeros((len(annots), 5))

        for idx, annot in enumerate(annots):
            class_name = annot['class']
            boxes[idx, 0] = float(annot['x1'])
            boxes[idx, 1] = float(annot['y1'])
            boxes[idx, 2] = float(annot['x2'])
            boxes[idx, 3] = float(annot['y2'])
            boxes[idx, 4] = self.name_to_label(class_name)

        return boxes
