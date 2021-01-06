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

from tensorflow.keras.layers import Conv2D, Input, Activation, Dropout, Reshape, Activation, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform,zeros
from tensorflow.keras import backend as K
import keras_retinanet.initializers
import keras_retinanet.layers
import keras_retinanet.losses
from tensorflow.keras.layers import BatchNormalization
import numpy as np
import keras_resnet

custom_objects = {
    'UpsampleLike': keras_retinanet.layers.UpsampleLike,
    'PriorProbability': keras_retinanet.initializers.PriorProbability,
    'RegressBoxes': keras_retinanet.layers.RegressBoxes,
    'NonMaximumSuppression': keras_retinanet.layers.NonMaximumSuppression,
    'Anchors': keras_retinanet.layers.Anchors,
    '_smooth_l1': keras_retinanet.losses.smooth_l1(),
    '_focal': keras_retinanet.losses.focal(),
    'BatchNormalization': keras_resnet.layers.BatchNormalization,
}


def default_classification_model(
        num_classes,
        num_anchors,
        pyramid_feature_size=64,
        prior_probability=0.01,
        classification_feature_size=64,
        name='classification_submodel'
):
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
    }

    inputs = Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = Conv2D(
            filters=classification_feature_size,
            # activation='relu',
            name='pyramid_classification_{}'.format(i),
            kernel_initializer=glorot_uniform(),
            bias_initializer='zeros',
            **options
        )(outputs)
        outputs = BatchNormalization(axis=-1, name='pyramid_classification_{}_bn'.format(i))(outputs)
        outputs = Activation('relu', name='pyramid_classification_{}_relu'.format(i))(outputs)
        outputs = Dropout(0.3)(outputs)

    outputs = Conv2D(
        filters=num_classes * num_anchors,
        kernel_initializer=zeros(),
        bias_initializer=keras_retinanet.initializers.PriorProbability(probability=prior_probability),
        name='pyramid_classification',
        **options
    )(outputs)

    # reshape output and apply sigmoid
    outputs = Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)
    return Model(inputs=inputs, outputs=outputs, name=name)


def default_regression_model(num_anchors, pyramid_feature_size=64, regression_feature_size=64,
                             name='regression_submodel'):
    # All new conv layers except the final one in the
    # RetinaNet (classification) subnets are initialized
    # with bias b = 0 and a Gaussian weight fill with stddev = 0.01.
    options = {
        'kernel_size': 3,
        'strides': 1,
        'padding': 'same',
        'kernel_initializer': glorot_uniform(),
        'bias_initializer': 'zeros'
    }

    inputs = Input(shape=(None, None, pyramid_feature_size))
    outputs = inputs
    for i in range(4):
        outputs = Conv2D(
            filters=regression_feature_size,
            # activation='relu',
            name='pyramid_regression_{}'.format(i),
            **options
        )(outputs)
        outputs = BatchNormalization(axis=-1, name='pyramid_regression_{}_bn'.format(i))(outputs)
        outputs = Activation('relu', name='pyramid_regression_{}_relu'.format(i))(outputs)
        outputs = Dropout(0.3)(outputs)

    outputs = Conv2D(num_anchors * 4, name='pyramid_regression', **options)(outputs)
    outputs = Reshape((-1, 4), name='pyramid_regression_reshape')(outputs)
    return Model(inputs=inputs, outputs=outputs, name=name)


def __create_pyramid_features(C3, C4, C5, feature_size=64):
    # upsample C5 to get P5 from the FPN paper
    P5 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='P5')(C5)
    P5_upsampled = keras_retinanet.layers.UpsampleLike(name='P5_upsampled')([P5, C4])

    # add P5 elementwise to C4
    P4 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    P4 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)
    P4_upsampled = keras_retinanet.layers.UpsampleLike(name='P4_upsampled')([P4, C3])

    # add P4 elementwise to C3
    P3 = Conv2D(feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D(feature_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    # "P6 is obtained via a 3x3 stride-2 conv on C5"
    P6 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
    P7 = Activation('relu', name='C6_relu')(P6)
    P7 = Conv2D(feature_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    return P3, P4, P5, P6, P7


class AnchorParameters:
    def __init__(self, sizes, strides, ratios, scales):
        self.sizes = sizes
        self.strides = strides
        self.ratios = ratios
        self.scales = scales

    def num_anchors(self):
        return len(self.ratios) * len(self.scales)


AnchorParameters.default = AnchorParameters(
    sizes=[32, 64, 128, 256, 512],
    strides=[8, 16, 32, 64, 128],
    ratios=np.array([0.5, 1, 2], K.floatx()),
    scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], K.floatx()),
)


def default_submodels(num_classes, anchor_parameters, feature_size=64):
    return [
        ('regression', default_regression_model(anchor_parameters.num_anchors(), pyramid_feature_size=feature_size,
                                                regression_feature_size=feature_size)),
        ('classification',
         default_classification_model(num_classes, anchor_parameters.num_anchors(), pyramid_feature_size=feature_size,
                                      classification_feature_size=feature_size))
    ]


def __build_model_pyramid(name, model, features):
    return Concatenate(axis=1, name=name)([model(f) for f in features])


def __build_pyramid(models, features):
    return [__build_model_pyramid(name, model, features) for name, model in models]


def __build_anchors(anchor_parameters, features):
    anchors = []
    for i, f in enumerate(features):
        anchors.append(keras_retinanet.layers.Anchors(
            size=anchor_parameters.sizes[i],
            stride=anchor_parameters.strides[i],
            ratios=anchor_parameters.ratios,
            scales=anchor_parameters.scales,
            name='anchors_{}'.format(i)
        )(f))
    return Concatenate(axis=1)(anchors)


def retinanet(
        inputs,
        backbone,
        num_classes,
        anchor_parameters=AnchorParameters.default,
        create_pyramid_features=__create_pyramid_features,
        feature_size=64,
        submodels=None,
        name='retinanet'
):
    if submodels is None:
        submodels = default_submodels(num_classes, anchor_parameters, feature_size)

    image = inputs

    _, C3, C4, C5 = backbone.outputs  # we ignore C2

    # compute pyramid features as per https://arxiv.org/abs/1708.02002
    features = create_pyramid_features(C3, C4, C5, feature_size)

    # for all pyramid levels, run available submodels
    pyramid = __build_pyramid(submodels, features)
    anchors = __build_anchors(anchor_parameters, features)

    return Model(inputs=inputs, outputs=[anchors] + pyramid, name=name)


def retinanet_bbox(inputs, num_classes, nms=True, name='retinanet-bbox', feature_size=64, *args, **kwargs):
    model = retinanet(inputs=inputs, num_classes=num_classes, feature_size=feature_size, *args, **kwargs)

    # we expect the anchors, regression and classification values as first output
    anchors = model.outputs[0]
    regression = model.outputs[1]
    classification = model.outputs[2]
    if len(model.outputs) > 3:
        other = Concatenate(axis=2, name='other')(model.outputs[2:])
    else:
        other = None

    # apply predicted regression to anchors
    boxes = keras_retinanet.layers.RegressBoxes(name='boxes')([anchors, regression])
    detections = Concatenate(axis=2)([boxes, classification] + ([other] if other is not None else []))

    # additionally apply non maximum suppression
    if nms:
        detections = keras_retinanet.layers.NonMaximumSuppression(name='nms')([boxes, classification, detections])

    # construct the model
    return Model(inputs=inputs, outputs=[regression, classification, detections], name=name)
