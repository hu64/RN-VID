"""
Copyright 2017-2018 cgratie (https://github.com/cgratie/)

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


import keras
from keras.utils import get_file
from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image
from keras import backend
from keras import engine
from keras import layers
from keras import models
import numpy as np
import tensorflow as tf
from keras import backend as K
from .. import layers as custom_layers


class VGGBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return vgg_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):
        """ Downloads ImageNet weights and returns path to weights file.
        Weights can be downloaded at https://github.com/fizyr/keras-models/releases .
        """
        if self.backbone == 'vgg16' \
                or self.backbone == 'vgg16_flow_s' \
                or self.backbone == 'vgg16_flow_y' \
                or self.backbone == 'vgg16_flow_3d' \
                or self.backbone == 'vgg16_flow_c' \
                or self.backbone == 'vgg16_sf'\
                or self.backbone == 'vgg16_sf_flow' \
                or self.backbone == 'vgg16_5f' \
                or self.backbone == 'vgg16_3f':
            resource = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
            checksum = '6d6bbae143d832006294945121d1f1fc'
        elif self.backbone == 'vgg19':
            resource = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
            checksum = '253f8cb515780f3b799900260a226db6'
        else:
            raise ValueError("Backbone '{}' not recognized.".format(self.backbone))

        return get_file(
            '{}_weights_tf_dim_ordering_tf_kernels_notop.h5'.format(self.backbone),
            resource,
            cache_subdir='models',
            file_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['vgg16', 'vgg19', 'vgg16_flow_s','vgg16_sf', 'vgg16_sf_flow', 'vgg16_flow_y', 'vgg16_flow_3d', 'vgg16_flow_c', 'vgg16_5f', 'vgg16_3f']

        if self.backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(self.backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def VGG16_flow_s(include_top=True,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # Determine proper input shape
    input_shape = (None, None, 6)

    #average or max pooling
    average_pooling = False

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)


    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='vgg16_flow_s')

    return model


def VGG16_flow_y(include_top=True,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # Determine proper input shape
    input_shape = None, None, 3, 2

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # split block
    # split = layers.Lambda(lambda x: tf.split(x, 2, axis=4), name='split')(img_input)
    # img_input1 = layers.Lambda(lambda x: keras.backend.squeeze(x, axis=4), name='squeeze1')(split[0])
    # img_input2 = layers.Lambda(lambda x: keras.backend.squeeze(x, axis=4), name='squeeze2')(split[1])

    def split_f(img_input, num_or_size_splits=2, axis=3):
        import keras.backend as K
        import tensorflow as tf
        return tf.split(img_input, num_or_size_splits, axis)
    split = layers.Lambda(split_f, name='split2d', arguments={'num_or_size_splits': 2, 'axis': 3})(img_input)
    img_input1 = split[0]
    img_input2 = split[1]

    # Block 1-1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1-1')(img_input1)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2-1')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1-1_pool')(x)

    # Block 2-1
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1-1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2-1')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2-1_pool')(x)

    # Block 1-2
    y = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1-2')(img_input2)
    y = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2-2')(y)
    y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1-2_pool')(y)

    # Block 2-2
    y = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1-2')(y)
    y = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2-2')(y)
    y = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2-2_pool')(y)

    # x = layers.Concatenate(axis=2)([x, y])
    x = layers.Concatenate()([x, y])

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name='vgg16_flow_y')

    return model


def VGG16_flow_3d(include_top=True,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000):
    """Instantiates the VGG16 architecture.

    Optionally loads weights pre-trained on ImageNet.
    Note that the data format convention used by the model is
    the one specified in your Keras config at `~/.keras/keras.json`.

    # Arguments
        include_top: whether to include the 3 fully-connected
            layers at the top of the network.
        weights: one of `None` (random initialization),
              'imagenet' (pre-training on ImageNet),
              or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor
            (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)`
            (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 input channels,
            and width and height should be no smaller than 48.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """
    # Determine proper input shape
    input_shape = (2, None, None, 3)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = layers.ConvLSTM2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)    #print(x.shape)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = models.Model(inputs, x, name='vgg16_flow_3d')

    return model



def VGG16_sf(include_top=True,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000):

    # Determine proper input shape
    input_shape = (None, None, 15)

    #average or max pooling
    average_pooling = False


    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(shape=input_shape)
        else:
            img_input = input_tensor


    x1 = custom_layers.Split1(name='split1')(img_input)
    x2 = custom_layers.Split2(name='split2')(img_input)
    x3 = custom_layers.Split3(name='split3')(img_input)
    x4 = custom_layers.Split4(name='split4')(img_input)
    x5 = custom_layers.Split5(name='split5')(img_input)

    # Block 1
    b1c1 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1', trainable=False)
    b1c2 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2', trainable=False)


    if average_pooling:
        b1p = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')
    else:
        b1p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')


    x1 = b1c1(x1)
    x1 = b1c2(x1)
    x1 = b1p(x1)

    x2 = b1c1(x2)
    x2 = b1c2(x2)
    x2 = b1p(x2)

    x3 = b1c1(x3)
    x3 = b1c2(x3)
    x3 = b1p(x3)

    x4 = b1c1(x4)
    x4 = b1c2(x4)
    x4 = b1p(x4)

    x5 = b1c1(x5)
    x5 = b1c2(x5)
    x5 = b1p(x5)

    """
    three_way_merge = True
    one_by_one_per_channel = True
    if one_by_one_per_channel:
        merge = custom_layers.OneByOneMergeConv3D()
        x11 = merge([x1, x2, x3])
        if three_way_merge:
            x12 = merge([x2, x3, x4])
        x13 = merge([x3, x4, x5])
    else:
        x11 = layers.Maximum()([x1, x2, x3])
        if three_way_merge:
            x12 = layers.Maximum()([x2, x3, x4])
        x13 = layers.Maximum()([x3, x4, x5])
    """
    # Block 2
    b2c1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1', trainable=False)
    b2c2 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2', trainable=False)

    if average_pooling:
        b2p = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')
    else:
        b2p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    x1 = b2c1(x1)
    x1 = b2c2(x1)
    x1 = b2p(x1)

    x2 = b2c1(x2)
    x2 = b2c2(x2)
    x2 = b2p(x2)

    x3 = b2c1(x3)
    x3 = b2c2(x3)
    x3 = b2p(x3)

    x4 = b2c1(x4)
    x4 = b2c2(x4)
    x4 = b2p(x4)

    x5 = b2c1(x5)
    x5 = b2c2(x5)
    x5 = b2p(x5)

    x21 = custom_layers.OneByOneMerge()([x1, x2, x3])
    x22 = custom_layers.OneByOneMerge()([x2, x3, x4])
    x23 = custom_layers.OneByOneMerge()([x3, x4, x5])

    # Block 3
    b3c1 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1', trainable=False)
    b3c2 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2', trainable=False)
    b3c3 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3', trainable=False)
    if average_pooling:
        b3p = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool_0')
    else:
        b3p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_0')

    x21 = b3c1(x21)
    x21 = b3c2(x21)
    x21 = b3c3(x21)
    x21 = b3p(x21)

    x22 = b3c1(x22)
    x22 = b3c2(x22)
    x22 = b3c3(x22)
    x22 = b3p(x22)

    x23 = b3c1(x23)
    x23 = b3c2(x23)
    x23 = b3c3(x23)
    x23 = b3p(x23)

    # x = custom_layers.OneByOneMergeConv3D(name='block3_pool')([x21, x22, x23])
    x = custom_layers.OneByOneMerge(name='block3_pool')([x21, x22, x23])
    # x = layers.Maximum(name='block3_pool')([x21, x22, x23])

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1', trainable=False)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2', trainable=False)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3', trainable=False)(x)
    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1', trainable=False)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2', trainable=False)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3', trainable=False)(x)
    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input


    # Create model.
    # model = models.Model(inputs, x, name='vgg16_sf')
    model = models.Model(img_input, x, name='vgg16_sf')

    return model

def VGG16_sf_flow(include_top=True,
          input_tensor=None,
          input_shape=None,
          pooling=None,
          classes=1000):


    # Determine proper input shape
    input_shape = (None, None, 11)

    #average or max pooling
    average_pooling = False


    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(shape=input_shape)
        else:
            img_input = input_tensor

    x2 = custom_layers.Split1(name='split2')(img_input)
    x3 = custom_layers.Split2(name='split3')(img_input)
    x4 = custom_layers.Split3(name='split4')(img_input)
    f1 = custom_layers.SplitFlow1(name='flow1')(img_input)
    f2 = custom_layers.SplitFlow2(name='flow2')(img_input)


    # Block 1
    b1c1 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')
    b1c2 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')


    if average_pooling:
        b1p = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block1_pool')
    else:
        b1p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

    x2 = b1c1(x2)
    x2 = b1c2(x2)
    x2 = b1p(x2)

    x3 = b1c1(x3)
    x3 = b1c2(x3)
    x3 = b1p(x3)

    x4 = b1c1(x4)
    x4 = b1c2(x4)
    x4 = b1p(x4)

    f1 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_flow_pool')(f1)
    f2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_flow_pool')(f2)

    x11 = layers.Maximum()([x2, x3])
    x11 = layers.Multiply()([x11, f1])


    x12 = layers.Maximum()([x3, x4])
    x12 = layers.Multiply()([x12, f2])

    # Block 2
    b2c1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')
    b2c2 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')

    if average_pooling:
        b2p = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block2_pool')
    else:
        b2p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    x11 = b2c1(x11)
    x11 = b2c2(x11)
    x11 = b2p(x11)

    x12 = b2c1(x12)
    x12 = b2c2(x12)
    x12 = b2p(x12)

    x = layers.Maximum()([x11, x12])

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    if average_pooling:
        x = layers.AveragePooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    else:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = engine.get_source_inputs(input_tensor)
    else:
        inputs = img_input


    # Create model.
    # model = models.Model(inputs, x, name='vgg16_sf')
    model = models.Model(img_input, x, name='vgg16_sf_flow')

    return model

def vgg_retinanet(num_classes, backbone='vgg16', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a vgg backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('vgg16', 'vgg19')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a VGG backbone.
    """

    if backbone == 'vgg16_5f':
        return vgg16_retinanet_5f(num_classes=num_classes, inputs=inputs, modifier=modifier, **kwargs)
    if backbone == 'vgg16_3f':
        return vgg16_retinanet_3f(num_classes=num_classes, inputs=inputs, modifier=modifier, **kwargs)
    # choose default input
    if inputs is None and '_sf' not in backbone:
        inputs = keras.layers.Input(shape=(None, None, 3))

    # create the vgg backbone
    if backbone == 'vgg16':
        vgg = keras.applications.VGG16(input_tensor=inputs, include_top=False)
        # weights = '/store/datasets/UAV/models/vgg16-rn-w-s-1-on/snapshots/vgg16_csv_14.h5'
        # weights = '/store/datasets/UA-Detrac/models2/vgg16-1-on/snapshots/vgg16_csv_07.h5'
        # vgg.load_weights(weights, by_name=True)
        # for layer in vgg.layers[:-4]:
        #     layer.trainable = False
    elif backbone == 'vgg19':
        vgg = keras.applications.VGG19(input_tensor=inputs, include_top=False)
    elif backbone == 'vgg16_flow_s':
        inputs = keras.layers.Input(shape=(None, None, 6))
        vgg = VGG16_flow_s(input_tensor=inputs, include_top=False)
    elif backbone == 'vgg16_flow_y':
        inputs = keras.layers.Input(shape=(None, None, 6))
        vgg = VGG16_flow_y(input_tensor=inputs, include_top=False)
    elif backbone == 'vgg16_flow_3d':
        inputs = keras.layers.Input(shape=(2, None, None, 3))
        vgg = VGG16_flow_3d(input_tensor=inputs, include_top=False)
    elif backbone == 'vgg16_sf':
        if inputs is None:
            inputs = keras.layers.Input(shape=(None, None, 15))
        vgg = VGG16_sf(input_tensor=inputs, include_top=False)
    elif backbone == 'vgg16_sf_flow':
        if inputs is None:
            inputs = keras.layers.Input(shape=(None, None, 11))
        vgg = VGG16_sf_flow(input_tensor=inputs, include_top=False)
    # elif backbone == 'vgg16_flow_c':
    #     inputs = keras.layers.Input(shape=(None, None, 6))
    #     vgg = VGG16_flow_c(input_tensor=inputs, include_top=False)
    else:
        raise ValueError("Backbone '{}' not recognized.".format(backbone))

    if modifier:
        vgg = modifier(vgg)

    # create the full model
    layer_names = ["block3_pool", "block4_pool", "block5_pool"]
    layer_outputs = [vgg.get_layer(name).output for name in layer_names]
    model = retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=layer_outputs, **kwargs)

    # model.save('/store/datasets/ILSVRC2015/models/rn-vgg16-sm256/model.h5')
    # exit(0)


    return model

def vgg16_retinanet_5f(num_classes, inputs=None, modifier=None, **kwargs):

    inputs = keras.layers.Input(shape=(None, None, 15))

    x1 = custom_layers.Split1(name='split1')(inputs)
    x2 = custom_layers.Split2(name='split2')(inputs)
    x3 = custom_layers.Split3(name='split3')(inputs)
    x4 = custom_layers.Split4(name='split4')(inputs)
    x5 = custom_layers.Split5(name='split5')(inputs)

    #

    """
    vgg1 = keras.applications.VGG16(input_tensor=x1, include_top=False, weights='imagenet')
    vgg1.load_weights(weights, by_name=True)
    vgg2 = keras.applications.VGG16(input_tensor=x2, include_top=False, weights='imagenet')
    vgg2.load_weights(weights, by_name=True)
    vgg3 = keras.applications.VGG16(input_tensor=x3, include_top=False, weights='imagenet')
    vgg3.load_weights(weights, by_name=True)
    vgg4 = keras.applications.VGG16(input_tensor=x4, include_top=False, weights='imagenet')
    vgg4.load_weights(weights, by_name=True)
    vgg5 = keras.applications.VGG16(input_tensor=x5, include_top=False, weights='imagenet')
    vgg5.load_weights(weights, by_name=True)
    """

    layer_outputs1 = []
    layer_outputs2 = []
    layer_outputs3 = []
    layer_outputs4 = []
    layer_outputs5 = []


    vgg1 = keras.applications.VGG16(input_tensor=x1, include_top=False, weights='imagenet')
    # vgg1.load_weights(weights, by_name=True)

    for layer in vgg1.layers[:-4]:
        layer.trainable = False

    for layer in vgg1.layers:
        if 'block3_pool' in layer.name:
            layer_outputs1.append(layer.output)

            layer2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_1')
            x2 = layer2(x2)
            layer_outputs2.append(layer2.output)

            layer3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_2')
            x3 = layer3(x3)
            layer_outputs3.append(layer3.output)

            layer4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_3')
            x4 = layer4(x4)
            layer_outputs4.append(layer4.output)

            layer5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_4')
            x5 = layer5(x5)
            layer_outputs5.append(layer5.output)
        elif 'block4_pool' in layer.name:
            layer_outputs1.append(layer.output)

            layer2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_1')
            x2 = layer2(x2)
            layer_outputs2.append(layer2.output)

            layer3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_2')
            x3 = layer3(x3)
            layer_outputs3.append(layer3.output)

            layer4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_3')
            x4 = layer4(x4)
            layer_outputs4.append(layer4.output)

            layer5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_4')
            x5 = layer5(x5)
            layer_outputs5.append(layer5.output)
        elif 'block5_pool' in layer.name:
            layer_outputs1.append(layer.output)

            layer2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_1')
            x2 = layer2(x2)
            layer_outputs2.append(layer2.output)

            layer3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_2')
            x3 = layer3(x3)
            layer_outputs3.append(layer3.output)

            layer4 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_3')
            x4 = layer4(x4)
            layer_outputs4.append(layer4.output)

            layer5 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_4')
            x5 = layer5(x5)
            layer_outputs5.append(layer5.output)
        elif 'block' in layer.name:
            x2 = layer(x2)
            x3 = layer(x3)
            x4 = layer(x4)
            x5 = layer(x5)

    """

    networks = [vgg1, vgg2, vgg3, vgg4, vgg5]

    for j, network in enumerate(networks):
        for layer in network.layers:
            layer.name += '_' + str(j)
        # for layer in network.layers[:-4]:
        for layer in network.layers:
            layer.trainable = False
    """

    if modifier:
        vgg1 = modifier(vgg1)
        # vgg2 = modifier(vgg2)
        # vgg3 = modifier(vgg3)
        # vgg4 = modifier(vgg4)
        # vgg5 = modifier(vgg5)

    # create the full model
    layer_names = ["block3_pool", "block4_pool", "block5_pool"]

    # layer_outputs1 = [vgg1.get_layer(name + '_0').output for name in layer_names]
    # layer_outputs2 = [vgg2.get_layer(name + '_1').output for name in layer_names]
    # layer_outputs3 = [vgg3.get_layer(name + '_2').output for name in layer_names]
    # layer_outputs4 = [vgg4.get_layer(name + '_3').output for name in layer_names]
    # layer_outputs5 = [vgg5.get_layer(name + '_4').output for name in layer_names]

    # layer_outputs1 = [vgg1.get_layer(name).output for name in layer_names]
    # layer_outputs2 = [vgg2.get_layer(name).output for name in layer_names]
    # layer_outputs3 = [vgg3.get_layer(name).output for name in layer_names]
    # layer_outputs4 = [vgg4.get_layer(name).output for name in layer_names]
    # layer_outputs5 = [vgg5.get_layer(name).output for name in layer_names]

    model = retinanet.retinanet_5f(inputs=inputs, num_classes=num_classes, backbone_layers=[layer_outputs1,
                                                                                            layer_outputs2,
                                                                                            layer_outputs3,
                                                                                            layer_outputs4,
                                                                                            layer_outputs5,], **kwargs)
    # model.save('/store/datasets/ILSVRC2015/models/5f_b/model.h5')
    # exit()

    weights = '/store/datasets/ILSVRC2015/models/5f/snapshots2/vgg16_5f_csv_07.h5'
    model.load_weights(weights, by_name=True)

    for layer in model.layers:
        print(layer.name)
    #exit()

    return model

def vgg16_retinanet_3f(num_classes, inputs=None, modifier=None, **kwargs):

    inputs = keras.layers.Input(shape=(None, None, 9))

    x1 = custom_layers.Split1(name='split1')(inputs)
    x2 = custom_layers.Split2(name='split2')(inputs)
    x3 = custom_layers.Split3(name='split3')(inputs)

    layer_outputs1 = []
    layer_outputs2 = []
    layer_outputs3 = []

    weights = '/store/datasets/UAV/models/vgg16-fbeb5/snapshots-pt/vgg16_csv_20.h5'
    vgg1 = keras.applications.VGG16(input_tensor=x1, include_top=False, weights='imagenet')
    vgg1.load_weights(weights, by_name=True)

    for layer in vgg1.layers[:-4]:
        layer.trainable = False

    for layer in vgg1.layers:
        if 'block3_pool' in layer.name:
            layer_outputs1.append(layer.output)

            layer2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_1')
            x2 = layer2(x2)
            layer_outputs2.append(layer2.output)

            layer3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_2')
            x3 = layer3(x3)
            layer_outputs3.append(layer3.output)
        elif 'block4_pool' in layer.name:
            layer_outputs1.append(layer.output)

            layer2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_1')
            x2 = layer2(x2)
            layer_outputs2.append(layer2.output)

            layer3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_2')
            x3 = layer3(x3)
            layer_outputs3.append(layer3.output)
        elif 'block5_pool' in layer.name:
            layer_outputs1.append(layer.output)

            layer2 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_1')
            x2 = layer2(x2)
            layer_outputs2.append(layer2.output)

            layer3 = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_2')
            x3 = layer3(x3)
            layer_outputs3.append(layer3.output)
        elif 'block' in layer.name:
            x2 = layer(x2)
            x3 = layer(x3)



    if modifier:
        vgg1 = modifier(vgg1)

    # create the full model
    layer_names = ["block3_pool", "block4_pool", "block5_pool"]

    model = retinanet.retinanet_3f(inputs=inputs, num_classes=num_classes, backbone_layers=[layer_outputs1,
                                                                                            layer_outputs2,
                                                                                            layer_outputs3], **kwargs)

    model.load_weights('/store/datasets/UAV/models/vgg16-fbeb5/snapshots-pt/vgg16_csv_20.h5', by_name=True)

    # model.save('/store/datasets/UAV/models/vgg16-3f-2D/model.h5')
    # exit()

    return model

def vgg16_retinanet_5f_0(num_classes, inputs=None, modifier=None, **kwargs):

    inputs = keras.layers.Input(shape=(None, None, 15))

    x1 = custom_layers.Split1(name='split1')(inputs)
    x2 = custom_layers.Split2(name='split2')(inputs)
    x3 = custom_layers.Split3(name='split3')(inputs)
    x4 = custom_layers.Split4(name='split4')(inputs)
    x5 = custom_layers.Split5(name='split5')(inputs)

    # layers:
    b1c1 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')
    b1c2 = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')
    b1p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')

    b2c1 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')
    b2c2 = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')
    b2p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    b3c1= layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')
    b3c2 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')
    b3c3 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')

    b4c1 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')
    b4c2 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')
    b4c3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')

    b5c1 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')
    b5c2 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')
    b5c3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')


    layer_outputs = []
    for i, frame in enumerate([x1, x2, x3, x4, x5]):
        layer_output = []

        x = b1c1(frame)
        x = b1c2(x)
        x = b1p(x)

        x = b2c1(x)
        x = b2c2(x)
        x = b2p(x)

        x = b3c1(x)
        x = b3c2(x)
        x = b3c3(x)
        b3p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_' + str(i))
        x = b3p(x)
        layer_output.append(b3p.output)

        x = b4c1(x)
        x = b4c2(x)
        x = b4c3(x)
        b4p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_' + str(i))
        x = b4p(x)
        layer_output.append(b4p.output)

        x = b5c1(x)
        x = b5c2(x)
        x = b5c3(x)
        b5p = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_' + str(i))
        x = b5p(x)
        layer_output.append(b5p.output)
        layer_outputs.append(layer_output)

    model = retinanet.retinanet_5f(inputs=inputs, num_classes=num_classes, backbone_layers=[layer_outputs[0],
                                                                                            layer_outputs[1],
                                                                                            layer_outputs[2],
                                                                                            layer_outputs[3],
                                                                                            layer_outputs[4],], **kwargs)

    if modifier:
        model = modifier(model)

    weights_path = keras.utils.get_file(
        'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
        ('https://github.com/fchollet/deep-learning-models/'
         'releases/download/v0.1/'
         'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'),
        cache_subdir='models',
        file_hash='6d6bbae143d832006294945121d1f1fc')

    model.load_weights(weights_path, by_name=True)

    for layer in model.layers:
        print(layer.name)


    return model
