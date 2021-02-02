"""
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

import keras
from .. import backend
from ..utils import anchors as utils_anchors
import keras.backend as K
import tensorflow as tf
import numpy as np


class Anchors(keras.layers.Layer):
    """ Keras layer for generating achors for a given shape.
    """

    def __init__(self, size, stride, ratios=None, scales=None, *args, **kwargs):
        """ Initializer for an Anchors layer.

        Args
            size: The base size of the anchors to generate.
            stride: The stride of the anchors to generate.
            ratios: The ratios of the anchors to generate (defaults to [0.5, 1, 2]).
            scales: The scales of the anchors to generate (defaults to [2^0, 2^(1/3), 2^(2/3)]).
        """
        self.size   = size
        self.stride = stride
        self.ratios = ratios
        self.scales = scales

        if ratios is None:
            self.ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
        elif isinstance(ratios, list):
            self.ratios  = np.array(ratios)
        if scales is None:
            #hughes
            # self.scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
            self.scales = np.array([2 ** 0, 1 / (2 ** (1.0 / 3.0)), 1 / (2 ** (2.0 / 3.0))], keras.backend.floatx()),
        elif isinstance(scales, list):
            self.scales  = np.array(scales)

        self.num_anchors = len(ratios) * len(scales)
        self.anchors     = keras.backend.variable(utils_anchors.generate_anchors(
            base_size=size,
            ratios=ratios,
            scales=scales,
        ))

        super(Anchors, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        features = inputs
        features_shape = keras.backend.shape(features)[:3]

        # generate proposals from bbox deltas and shifted anchors
        anchors = backend.shift(features_shape[1:3], self.stride, self.anchors)
        anchors = keras.backend.tile(keras.backend.expand_dims(anchors, axis=0), (features_shape[0], 1, 1))

        return anchors

    def compute_output_shape(self, input_shape):
        if None not in input_shape[1:]:
            total = np.prod(input_shape[1:3]) * self.num_anchors
            return (input_shape[0], total, 4)
        else:
            return (input_shape[0], None, 4)

    def get_config(self):
        config = super(Anchors, self).get_config()
        config.update({
            'size'   : self.size,
            'stride' : self.stride,
            'ratios' : self.ratios.tolist(),
            'scales' : self.scales.tolist(),
        })
        return config


class Identity(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs

def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = 1 if keras.backend.image_data_format() == "channels_first" else -1
    filters = init.shape.as_list()[channel_axis]
    se_shape = (1, 1, filters)

    se = keras.layers.GlobalAveragePooling2D()(init)
    se = keras.layers.Reshape(se_shape)(se)
    se = keras.layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = keras.layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if keras.backend.image_data_format() == 'channels_first':
        se = keras.layers.Permute((3, 1, 2))(se)

    x = keras.layers.multiply([init, se])
    return x

class OneByOneMergeConv3D(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        t1, t2, t3 = inputs

        # concatenation = keras.layers.Concatenate()
        layer_list = []
        for i in range(t1.shape[3]):
            layer_list.append(t1[:, :, :, i:i+1])
            layer_list.append(t2[:, :, :, i:i+1])
            layer_list.append(t3[:, :, :, i:i+1])

        tensor =  keras.backend.stack(layer_list, axis=-1)
        SE = True
        if SE:
            tensor = keras.backend.squeeze(tensor, axis=-2)
            tensor = squeeze_excite_block(tensor)
            tensor = keras.backend.expand_dims(tensor, axis=-2)

        tensor = keras.layers.Conv3D(int(t1.shape[3]), (1, 1, 3), strides=(1, 1, 3), activation='relu', padding='same')(tensor)

        tensor = keras.backend.squeeze(tensor, axis=-2)
        return tensor

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3]

    def get_config(self):
        config = super(OneByOneMergeConv3D, self).get_config()
        return config

class OneByOneMerge5(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(OneByOneMerge5, self).__init__(**kwargs)
        self.trainable = True

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel',
                                       shape=(1, 5, 1, int(int(input_shape[0][-1])*5), int(input_shape[0][-1])),
                                       initializer='uniform',
                                       trainable=True)
        super(OneByOneMerge5, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        t1, t2, t3, t4, t5 = inputs
        # concatenation = keras.layers.Concatenate()
        layer_list = []
        for i in range(t1.shape[3]):
            layer_list.append(t1[:, :, :, i:i+1])
            layer_list.append(t2[:, :, :, i:i+1])
            layer_list.append(t3[:, :, :, i:i+1])
            layer_list.append(t4[:, :, :, i:i+1])
            layer_list.append(t5[:, :, :, i:i+1])


        tensor =  keras.backend.stack(layer_list, axis=-1)
        # print(self.kernel.shape)
        tensor = K.conv3d(tensor, self.kernel, padding='same', strides=(1, 1, 5))
        # print(tensor.shape)

        # tensor = tf.layers.conv3d(tensor, t1.shape[3], (1, 1, 5), activation='relu', strides=(1, 1, 5), padding='same')
        # conv3D_layer = keras.layers.Conv3D(int(t1.shape[3]), (1, 1, 5), strides=(1, 1, 5), activation='relu', padding='same')
        # conv3D_layer.set_weights = self.add_weight(name='kernel', shape=(1, 1, 5, int(t1.shape[3])), initializer='uniform', trainable=True)
        # tensor = conv3D_layer(tensor)

        tensor = keras.backend.squeeze(tensor, axis=-2)
        return tensor


    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3]

    def get_config(self):
        config = super(OneByOneMerge5, self).get_config()
        return config

class OneByOneMerge5_2D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(OneByOneMerge5_2D, self).__init__(**kwargs)
        self.trainable = True

    def build(self, input_shape):

        # """
        self.kernel  = []
        for i in range(input_shape[0][-1]):
            self.kernel.append(self.add_weight(name='kernel',
                                      shape=(1, 1, 5, 1),
                                      initializer='uniform',
                                      trainable=True))
        # """
        """
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1, 1, 5, 1),
                                      initializer='uniform',
                                      trainable=True)
        """

        super(OneByOneMerge5_2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        t1, t2, t3, t4, t5 = inputs

        concatenation = keras.layers.Concatenate()
        layer_list = []
        for i in range(t1.shape[3]):
            tensor = concatenation([t1[:, :, :, i:i + 1],
                                    t2[:, :, :, i:i + 1],
                                    t3[:, :, :, i:i + 1],
                                    t4[:, :, :, i:i + 1],
                                    t5[:, :, :, i:i + 1]])

            layer_list.append(K.conv2d(tensor, self.kernel[i], padding='same'))
            # layer_list.append(K.conv2d(tensor, self.kernel, padding='same'))

        outputs = concatenation(layer_list)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3]

class OneByOneMerge3_2D(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(OneByOneMerge3_2D, self).__init__(**kwargs)
        self.trainable = True

    def build(self, input_shape):

        # """
        self.kernel  = []
        for i in range(input_shape[0][-1]):
            self.kernel.append(self.add_weight(name='kernel',
                                      shape=(1, 1, 3, 1),
                                      initializer='uniform',
                                      trainable=True))
        # """

        super(OneByOneMerge3_2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        t1, t2, t3 = inputs

        concatenation = keras.layers.Concatenate()
        layer_list = []
        for i in range(t1.shape[3]):
            tensor = concatenation([t1[:, :, :, i:i + 1],
                                    t2[:, :, :, i:i + 1],
                                    t3[:, :, :, i:i + 1]])

            layer_list.append(K.conv2d(tensor, self.kernel[i], padding='same'))
            # layer_list.append(K.conv2d(tensor, self.kernel, padding='same'))

        outputs = concatenation(layer_list)
        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3]


class OneByOneMerge(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(OneByOneMerge, self).__init__(**kwargs)
        self.trainable = True

    def build(self, input_shape):
        self.kernel = []
        for i in range(input_shape[0][-1]):
            self.kernel.append(self.add_weight(name='kernel',
                                               shape=(1, 1, 3, 1),
                                               initializer='uniform',
                                               trainable=True))
        super(OneByOneMerge, self).build(input_shape)  # Be sure to call this at the end

    def call(self, inputs, **kwargs):
        t1, t2, t3 = inputs

        concatenation = keras.layers.Concatenate()
        layer_list = []
        for i in range(t1.shape[3]):
            tensor = concatenation([t1[:, :, :, i:i+1],
                                    t2[:, :, :, i:i+1],
                                    t3[:, :, :, i:i+1]])
            layer_list.append(K.conv2d(tensor, self.kernel[i]))

        outputs = concatenation(layer_list)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3]

class OneByOneMergeTwo(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        import tensorflow as tf

        t1, t2 = inputs

        conv_1by1 = keras.layers.Conv2D(1, (1, 1),
                      activation='relu',
                      padding='same',
                      name='block2_conv1x1',
                      kernel_initializer='glorot_uniform')
        concatenation = keras.layers.Concatenate()
        layer_list = []
        for i in range(t1.shape[3]):
            layer_list.append(conv_1by1(concatenation([t1[:, :, :, i:i + 1],
                                                       t2[:, :, :, i:i + 1]])))
        outputs = concatenation(layer_list)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0][0], input_shape[0][1], input_shape[0][2], input_shape[0][3]

class MaxPoolChannelWise(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        import tensorflow as tf

        layer_list = []
        for i in range(0, inputs.shape[3], 3):

            layer = keras.layers.Maximum()([inputs[:, :, :, i:i+1],
                                                      inputs[:, :, :, i+1:i+2],
                                                      inputs[:, :, :, i+2:i+3]])
            layer_list.append(layer)
        outputs = tf.concat(values=layer_list, axis=-1)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], int(input_shape[3]/3)

class SplitFlow1(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs[:, :, :, 9, None]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], 1

class SplitFlow2(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs[:, :, :, 10, None]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], 1

class Split1(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs[:, :, :, 0:3]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], 3

    def get_config(self):
        config = super(Split1, self).get_config()
        return config

class Split2(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs[:, :, :, 3:6]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], 3

    def get_config(self):
        config = super(Split2, self).get_config()
        return config

class Split3(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs[:, :, :, 6:9]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], 3

    def get_config(self):
        config = super(Split3, self).get_config()
        return config

class Split4(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs[:, :, :, 9:12]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], 3

    def get_config(self):
        config = super(Split4, self).get_config()
        return config

class Split5(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs[:, :, :, 12:15]

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[1], input_shape[2], 3

    def get_config(self):
        config = super(Split5, self).get_config()
        return config

class SE(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        import tensorflow as tf
        from keras import backend as K

        nb_channel = K.int_shape(inputs)[-1]

        x = keras.layers.GlobalAveragePooling2D()(inputs)
        x = keras.layers.Dense(nb_channel // 16, activation='relu')(x)
        x = keras.layers.Dense(nb_channel, activation='sigmoid')(x)

        x = keras.layers.Multiply()([inputs, x])

        return x

class DivideByFour(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs * 0.25

class DivideByTwo(keras.layers.Layer):

    def call(self, inputs, **kwargs):
        return inputs * 0.5

class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = keras.backend.shape(target)
        return backend.resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class RegressBoxes(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes.
    """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.

        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std  = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return backend.bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist(),
        })

        return config


class ClipBoxes(keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """

    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = keras.backend.cast(keras.backend.shape(image), keras.backend.floatx())

        x1 = backend.clip_by_value(boxes[:, :, 0], 0, shape[2])
        y1 = backend.clip_by_value(boxes[:, :, 1], 0, shape[1])
        x2 = backend.clip_by_value(boxes[:, :, 2], 0, shape[2])
        y2 = backend.clip_by_value(boxes[:, :, 3], 0, shape[1])

        return keras.backend.stack([x1, y1, x2, y2], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]
