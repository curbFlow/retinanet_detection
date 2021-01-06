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

import tensorflow.keras.backend as K
import keras_retinanet.backend


def bbox_transform_inv(boxes, deltas):
    boxes = K.reshape(boxes, (-1, 4))
    deltas = K.reshape(deltas, (-1, 4))

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = ctr_x + dx * widths
    pred_ctr_y = ctr_y + dy * heights
    pred_w = K.exp(dw) * widths
    pred_h = K.exp(dh) * heights

    pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
    pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
    pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
    pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

    pred_boxes = K.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=1)
    pred_boxes = K.expand_dims(pred_boxes, axis=0)

    return pred_boxes


def shift(shape, stride, anchors):
    """
    Produce shifted anchors based on shape of the map and stride size
    """
    shift_x = (K.arange(0, shape[1], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride
    shift_y = (K.arange(0, shape[0], dtype=K.floatx()) + K.constant(0.5, dtype=K.floatx())) * stride

    shift_x, shift_y = keras_retinanet.backend.meshgrid(shift_x, shift_y)
    shift_x = K.reshape(shift_x, [-1])
    shift_y = K.reshape(shift_y, [-1])

    shifts = K.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts = K.transpose(shifts)
    number_of_anchors = K.shape(anchors)[0]

    k = K.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = K.reshape(anchors, [1, number_of_anchors, 4]) + K.cast(K.reshape(shifts, [k, 1, 4]), K.floatx())
    shifted_anchors = K.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors
