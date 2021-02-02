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
from keras.utils import get_file
import keras_resnet
import keras_resnet.models

from . import retinanet
from . import Backbone
from ..utils.image import preprocess_image
from .. import layers as custom_layers


class ResNetBackbone(Backbone):
    """ Describes backbone information and provides utility functions.
    """

    def __init__(self, backbone):
        super(ResNetBackbone, self).__init__(backbone)
        self.custom_objects.update(keras_resnet.custom_objects)

    def retinanet(self, *args, **kwargs):
        """ Returns a retinanet model using the correct backbone.
        """
        return resnet_retinanet(*args, backbone=self.backbone, **kwargs)

    def download_imagenet(self):

        print("download here")
        """ Downloads ImageNet weights and returns path to weights file.
        """
        resnet_filename = 'ResNet-{}-model.keras.h5'
        resnet_resource = 'https://github.com/fizyr/keras-models/releases/download/v0.0.1/{}'.format(resnet_filename)
        depth = int(self.backbone.replace('-flow', '').replace('-5f', '').replace('resnet', ''))

        filename = resnet_filename.format(depth)
        resource = resnet_resource.format(depth)
        if depth == 50:
            checksum = '3e9f4e4f77bbe2c9bec13b53ee1c2319'
        elif depth == 101:
            checksum = '05dc86924389e5b401a9ea0348a3213c'
        elif depth == 152:
            checksum = '6ee11ef2b135592f8031058820bb9e71'

        return get_file(
            filename,
            resource,
            cache_subdir='models',
            md5_hash=checksum
        )

    def validate(self):
        """ Checks whether the backbone string is correct.
        """
        allowed_backbones = ['resnet50', 'resnet50-5f', 'resnet101', 'resnet101-flow', 'resnet152', 'resnet50-flow', 'resnet152-flow']
        backbone = self.backbone.split('_')[0]

        if backbone not in allowed_backbones:
            raise ValueError('Backbone (\'{}\') not in allowed backbones ({}).'.format(backbone, allowed_backbones))

    def preprocess_image(self, inputs):
        """ Takes as input an image and prepares it for being passed through the network.
        """
        return preprocess_image(inputs, mode='caffe')


def resnet_retinanet(num_classes, backbone='resnet50', inputs=None, modifier=None, **kwargs):
    """ Constructs a retinanet model using a resnet backbone.

    Args
        num_classes: Number of classes to predict.
        backbone: Which backbone to use (one of ('resnet50', 'resnet101', 'resnet152')).
        inputs: The inputs to the network (defaults to a Tensor of shape (None, None, 3)).
        modifier: A function handler which can modify the backbone before using it in retinanet (this can be used to freeze backbone layers for example).

    Returns
        RetinaNet model with a ResNet backbone.
    """
    # choose default input
    if inputs is None:
        if '5f' not in backbone:
            inputs = keras.layers.Input(shape=(None, None, 3))
        else:
            inputs = keras.layers.Input(shape=(None, None, 15))


    # create the resnet backbone
    if backbone == 'resnet50' or backbone == 'resnet50-flow':
        resnet = keras_resnet.models.ResNet50(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet50-5f':
        x1 = custom_layers.Split1(name='split1')(inputs)
        x2 = custom_layers.Split2(name='split2')(inputs)
        x3 = custom_layers.Split3(name='split3')(inputs)
        x4 = custom_layers.Split4(name='split4')(inputs)
        x5 = custom_layers.Split5(name='split5')(inputs)

        outputs = [[], [], [], [], []]

        # model = keras_resnet.models.ResNet50(keras.layers.Input(shape=(None, None, 3)), include_top=False, freeze_bn=True)

        # resnet1 = keras.applications.ResNet50(input_tensor=x1, include_top=False, weights='imagenet')
        # resnet2 = keras.applications.ResNet50(input_tensor=x2, include_top=False, weights='imagenet')
        # resnet3 = keras.applications.ResNet50(input_tensor=x3, include_top=False, weights='imagenet')
        # resnet4 = keras.applications.ResNet50(input_tensor=x4, include_top=False, weights='imagenet')
        # resnet5 = keras.applications.ResNet50(input_tensor=x5, include_top=False, weights='imagenet')

        resnet1 = keras.applications.ResNet50(input_tensor=x1, include_top=False, weights='imagenet')
        resnet2 = keras.applications.ResNet50(input_tensor=x2, include_top=False, weights='imagenet')
        resnet3 = keras.applications.ResNet50(input_tensor=x3, include_top=False, weights='imagenet')
        resnet4 = keras.applications.ResNet50(input_tensor=x4, include_top=False, weights='imagenet')
        resnet5 = keras.applications.ResNet50(input_tensor=x5, include_top=False, weights='imagenet')


        """
        for layer in model.layers:
            print(layer.name)
        for layer in model.layers:
            print(layer.name)
            x1 = layer(x1)
            x2 = layer(x2)
            x3 = layer(x3)
            x4 = layer(x4)
            x5 = layer(x5)

            if layer.name in outputs_names:
                outputs[0].append(x1)
                outputs[1].append(x2)
                outputs[2].append(x3)
                outputs[3].append(x4)
                outputs[4].append(x5)
        """

        # model2 = keras_resnet.models.ResNet50(x2, include_top=False, freeze_bn=True)
        # model3 = keras_resnet.models.ResNet50(x3, include_top=False, freeze_bn=True)
        # model4 = keras_resnet.models.ResNet50(x4, include_top=False, freeze_bn=True)
        # model5 = keras_resnet.models.ResNet50(x5, include_top=False, freeze_bn=True)

        # model1 = keras.applications.VGG16(input_tensor=x1, include_top=False, weights='imagenet')
        # model2 = keras.applications.VGG16(input_tensor=x2, include_top=False, weights='imagenet')
        # model3 = keras.applications.VGG16(input_tensor=x3, include_top=False, weights='imagenet')
        # model4 = keras.applications.VGG16(input_tensor=x4, include_top=False, weights='imagenet')
        # model5 = keras.applications.VGG16(input_tensor=x5, include_top=False, weights='imagenet')

        if modifier:
            # model = modifier(model)
            resnet1 = modifier(resnet1)
            resnet2 = modifier(resnet2)
            resnet3 = modifier(resnet3)
            resnet4 = modifier(resnet4)
            resnet5 = modifier(resnet5)
            # model1 = modifier(model1)
            # model2 = modifier(model2)
            # model3 = modifier(model3)
            # model4 = modifier(model4)
            # model5 = modifier(model5)
        base_numbers = [22, 40, 49]
        outputs_names = ['activation_' + str(base_numbers[0]),
                         'activation_' + str(base_numbers[1]),
                         'activation_' + str(base_numbers[2])]
        # outputs_names = ['res3d_relu', 'res4f_relu', 'res5c_relu']
        for layer in resnet1.layers:
            # print(layer.name)
            if layer.name in outputs_names:
                outputs[0].append(layer)
        base_numbers = [item + 49 for item in base_numbers]
        outputs_names = ['activation_' + str(base_numbers[0]),
                         'activation_' + str(base_numbers[1]),
                         'activation_' + str(base_numbers[2])]
        for layer in resnet2.layers:
            if layer.name in outputs_names:
                outputs[1].append(layer)
        base_numbers = [item + 49 for item in base_numbers]
        outputs_names = ['activation_' + str(base_numbers[0]),
                         'activation_' + str(base_numbers[1]),
                         'activation_' + str(base_numbers[2])]
        for layer in resnet3.layers:
            if layer.name in outputs_names:
                outputs[2].append(layer)
        base_numbers = [item + 49 for item in base_numbers]
        outputs_names = ['activation_' + str(base_numbers[0]),
                         'activation_' + str(base_numbers[1]),
                         'activation_' + str(base_numbers[2])]
        for layer in resnet4.layers:
            if layer.name in outputs_names:
                outputs[3].append(layer)
        base_numbers = [item + 49 for item in base_numbers]
        outputs_names = ['activation_' + str(base_numbers[0]),
                         'activation_' + str(base_numbers[1]),
                         'activation_' + str(base_numbers[2])]
        for layer in resnet5.layers:
            if layer.name in outputs_names:
                outputs[4].append(layer)

        # outputs.append(resnet1.outputs)
        # outputs.append(resnet2.outputs)
        # outputs.append(resnet3.outputs)
        # outputs.append(resnet4.outputs)
        # outputs.append(resnet5.outputs)

        model = retinanet.retinanet_5f(inputs=inputs, num_classes=num_classes, backbone_layers=outputs, **kwargs)
        return model

    elif backbone == 'resnet101' or backbone == 'resnet101-flow':
        resnet = keras_resnet.models.ResNet101(inputs, include_top=False, freeze_bn=True)
    elif backbone == 'resnet152' or backbone == 'resnet152-flow':
        resnet = keras_resnet.models.ResNet152(inputs, include_top=False, freeze_bn=True)
    else:
        raise ValueError('Backbone (\'{}\') is invalid.'.format(backbone))

    # invoke modifier if given
    if modifier:
        resnet = modifier(resnet)

    # a enlever:
    print('resnet outputs: ', resnet.outputs[1:])
    # create the full model
    model = retinanet.retinanet(inputs=inputs, num_classes=num_classes, backbone_layers=resnet.outputs[1:], **kwargs)

    # model.save(str('/store/datasets/aic19/track3/models/resnet152-rn-pt-s3/model.h5'))
    # exit(0)

    return model


def resnet50_retinanet(num_classes, inputs=None, **kwargs):
    model = resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)

    # model.save('/store/datasets/UAV/models/resnet50-rn-nw/model.h5')
    # exit(0)

    return model

def resnet50_retinanet_5f(num_classes, inputs=None, **kwargs):

    model = resnet_retinanet(num_classes=num_classes, backbone='resnet50', inputs=inputs, **kwargs)

    # model.save('/store/datasets/UAV/models/resnet50-rn-nw/model.h5')
    # exit(0)

    return model

def resnet101_retinanet(num_classes, inputs=None, **kwargs):

    model = resnet_retinanet(num_classes=num_classes, backbone='resnet101', inputs=inputs, **kwargs)

    # model.save('/store/datasets/aic19/track3/models/resnet101-rn-s1-1/model.h5')
    # exit(0)

    return model


def resnet152_retinanet(num_classes, inputs=None, **kwargs):

    model = resnet_retinanet(num_classes=num_classes, backbone='resnet152', inputs=inputs, **kwargs)
    # model.save('/store/datasets/aic19/track3/models/resnet152-rn-s1-1/model.h5')
    # exit(0)

    return model
