import os
import sys
import glob
import random
import math
import datetime
import itertools
import json
import re
import cv2
import scipy.io as sio
import logging
from collections import OrderedDict
import numpy as np
import scipy.misc
import scipy.ndimage
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
from keras.activations import softmax
import skimage.transform
from parsing_rcnn_model import *
from flow_warp import flow_warp
import utils

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')

class BatchNorm(KL.BatchNormalization):
    """Batch Normalization class. Subclasses the Keras BN class and
    hardcodes training=False so the BN layer doesn't update
    during training.

    Batch normalization has a negative effect on training if batches are small
    so we disable it here.
    """
    def __init__(self, training=False, **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.training = training
    def call(self, inputs, training=None):
        return super(self.__class__, self).call(inputs, training=self.training)


def identity_block_share(input_tensor_list, kernel_size, filters, stage, block,
                   use_bias=True):
    """The identity_block is the block that has no conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    conv1 = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a', 
                  use_bias=use_bias)
    bn1 = BatchNorm(axis=-1, name=bn_name_base + '2a')
    conv2 = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', 
                  name=conv_name_base + '2b', use_bias=use_bias)
    bn2 = BatchNorm(axis=-1, name=bn_name_base + '2b')
    conv3 = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',
                  use_bias=use_bias)
    bn3 = BatchNorm(axis=-1, name=bn_name_base + '2c')

    features = []
    for input_tensor in input_tensor_list:

        x = conv1(input_tensor)
        x = bn1(x)
        x = KL.Activation('relu')(x)

        x = conv2(x)
        x = bn2(x)
        x = KL.Activation('relu')(x)

        x = conv3(x)
        x = bn3(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu')(x)
        features.append(x)
    return features


def conv_block_share(input_tensor_list, kernel_size, filters, stage, block,
               strides=(2, 2), use_bias=True):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor_list: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    conv1 = KL.Conv2D(nb_filter1, (1, 1), strides=strides, 
                  name=conv_name_base + '2a', use_bias=use_bias)
    bn1 = BatchNorm(axis=-1, name=bn_name_base + '2a')

    conv2 = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', 
                  name=conv_name_base + '2b', use_bias=use_bias)
    bn2 = BatchNorm(axis=-1, name=bn_name_base + '2b')

    conv3 = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c', 
                  use_bias=use_bias)
    bn3 = BatchNorm(axis=-1, name=bn_name_base + '2c')

    conv4 = KL.Conv2D(nb_filter3, (1, 1), strides=strides, 
                  name=conv_name_base + '1', use_bias=use_bias)
    bn4 = BatchNorm(axis=-1, name=bn_name_base + '1')


    features = []

    for input_tensor in input_tensor_list:

        x = conv1(input_tensor)
        x = bn1(x)
        x = KL.Activation('relu')(x)

        x = conv2(x)
        x = bn2(x)
        x = KL.Activation('relu')(x)

        x = conv3(x)
        x = bn3(x)

        shortcut = conv4(input_tensor)
        shortcut = bn4(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu')(x)

        features.append(x)
    return features

# Atrous-Convolution version of residual blocks
def atrous_identity_block_share(input_tensor_list, kernel_size, filters, stage,
                          block, atrous_rate=(2, 2), use_bias=True):
    '''The identity_block is the block that has no conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    conv1 = KL.Conv2D(nb_filter1, (1, 1), name=conv_name_base + '2a',  
                  use_bias=use_bias)
    bn1 = BatchNorm(axis=-1, name=bn_name_base + '2a')

    conv2 = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), dilation_rate=atrous_rate,  
                      padding='same', name=conv_name_base + '2b', use_bias=use_bias)
    bn2 = BatchNorm(axis=-1, name=bn_name_base + '2b')

    conv3 = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',  
                  use_bias=use_bias)
    bn3 = BatchNorm(axis=-1, name=bn_name_base + '2c')

    features = []
    for input_tensor in input_tensor_list:

        x = conv1(input_tensor)
        x = bn1(x)
        x = KL.Activation('relu')(x)

        x = conv2(x)
        x = bn2(x)
        x = KL.Activation('relu')(x)

        x = conv3(x)
        x = bn3(x)

        x = KL.Add()([x, input_tensor])
        x = KL.Activation('relu')(x)
        features.append(x)
    return features

def atrous_conv_block_share(input_tensor_list, kernel_size, filters, stage, 
                     block, strides=(1, 1), atrous_rate=(2, 2), use_bias=True):
    '''conv_block is the block that has a conv layer at shortcut
    # Arguments
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    '''
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    conv1 = KL.Conv2D(nb_filter1, (1, 1), strides=strides, 
                  name=conv_name_base + '2a', use_bias=use_bias)
    bn1 = BatchNorm(axis=-1, name=bn_name_base + '2a')

    conv2 = KL.Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', dilation_rate=atrous_rate, 
                  name=conv_name_base + '2b', use_bias=use_bias)
    bn2 = BatchNorm(axis=-1, name=bn_name_base + '2b')

    conv3 = KL.Conv2D(nb_filter3, (1, 1), name=conv_name_base + '2c',  
                  use_bias=use_bias)
    bn3 = BatchNorm(axis=-1, name=bn_name_base + '2c')

    conv4 = KL.Conv2D(nb_filter3, (1, 1), strides=strides,  
                        name=conv_name_base + '1', use_bias=use_bias)
    bn4 = BatchNorm(axis=-1, name=bn_name_base + '1')


    features = []
    for input_tensor in input_tensor_list:

        x = conv1(input_tensor)
        x = bn1(x)
        x = KL.Activation('relu')(x)

        x = conv2(x)
        x = bn2(x)
        x = KL.Activation('relu')(x)

        x = conv3(x)
        x = bn3(x)

        shortcut = conv4(input_tensor)
        shortcut = bn4(shortcut)

        x = KL.Add()([x, shortcut])
        x = KL.Activation('relu')(x)
        features.append(x)
    return features

def deeplab_resnet_share(img_inputs, architecture):
    """
    Build the architecture of resnet-101.
    img_inputs: a list of input image
    """

    # Stage 1
    conv1 = KL.Conv2D(64, (7, 7), strides=(2, 2), 
        name='conv1', use_bias=False)
    bn_conv1 = BatchNorm(axis=-1, name='bn_conv1')
    c1 = []
    for img_input in img_inputs:
        x = KL.ZeroPadding2D((3, 3))(img_input)
        x = conv1(x)
        x = bn_conv1(x)
        x = KL.Activation('relu')(x)
        x = KL.MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
        c1.append(x)

    # Stage 2
    features = conv_block_share(c1, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1), use_bias=False)
    features = identity_block_share(features, 3, [64, 64, 256], stage=2, block='b', use_bias=False)
    features = identity_block_share(features, 3, [64, 64, 256], stage=2, block='c', use_bias=False)
    # Stage 3
    features = conv_block_share(features, 3, [128, 128, 512], stage=3, block='a', use_bias=False)
    features = identity_block_share(features, 3, [128, 128, 512], stage=3, block='b1', use_bias=False)
    features = identity_block_share(features, 3, [128, 128, 512], stage=3, block='b2', use_bias=False)
    features = identity_block_share(features, 3, [128, 128, 512], stage=3, block='b3', use_bias=False)
    # Stage 4
    features = conv_block_share(features, 3, [256, 256, 1024], stage=4, block='a', use_bias=False)
    block_count = {"resnet50": 5, "resnet101": 22}[architecture]
    for i in range(1, block_count+1):
        features = identity_block_share(features, 3, [256, 256, 1024], stage=4, block='b%d'%i, use_bias=False)
    # Stage 5
    features = atrous_conv_block_share(features, 3, [512, 512, 2048], stage=5, block='a', atrous_rate=(2, 2), use_bias=False)
    features = atrous_identity_block_share(features, 3, [512, 512, 2048], stage=5, block='b', atrous_rate=(2, 2), use_bias=False)
    c5 = atrous_identity_block_share(features, 3, [512, 512, 2048], stage=5, block='c', atrous_rate=(2, 2), use_bias=False)

    return c1, c5


def load_image_gt(dataset, config, image_id, augment=False,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: If true, apply random image augmentation. Currently, only
        horizontal flipping is offered.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    current_frame: [height, width, 3]
    adjcent_key_frames: p [height, width, 3]
    identity_ind: int, if current_frame is a key frame, identity_ind is setted 0 else 1
    image_meta: 
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    part: [height, width]
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    keys, identity_ind = dataset.load_keys(image_id, config.KEY_RANGE_L, 3) 
    mask, class_ids = dataset.load_mask(image_id)
    part = dataset.load_part(image_id)
    part_rev = dataset.load_reverse_part(image_id)
    shape = image.shape

    image, window, scale, padding = utils.resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    mask = utils.resize_mask(mask, scale, padding)
    part = utils.resize_part(part, scale, padding[:2])
    part_rev = utils.resize_part(part_rev, scale, padding[:2])

    # process nearby frame
    h, w = keys[0].shape[:2]
    for i in range(len(keys)):
        keys[i] = scipy.misc.imresize(keys[i], (round(h * scale), round(w * scale)))
        keys[i] = np.pad(keys[i], padding, mode='constant', constant_values=0)

    # Random horizontal flips.
    if augment:
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)
            part = part_rev
            for i in range(len(keys)):
                keys[i] = np.fliplr(keys[i])

    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = utils.extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, shape, window, active_class_ids)
    return [image,]+keys+[np.array([identity_ind], dtype=np.int32),image_meta, class_ids, bbox, mask, part]


def data_generator(dataset, config, shuffle=True, augment=True, random_rois=0,
                   batch_size=1, detection_targets=False):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
             horizontal flips are supported)
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.
    - gt_parts: [batch, height, width] of uint8 type. The height and width are
                those of the image

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    anchors = utils.generate_anchors(config.RPN_ANCHOR_SCALES,
                                     config.RPN_ANCHOR_RATIOS,
                                     config.BACKBONE_SHAPES[0],
                                     config.BACKBONE_STRIDES[0],
                                     config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, key1, key2, key3, identity_ind, image_meta, gt_class_ids, gt_boxes, gt_masks, gt_parts = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              use_mini_mask=config.USE_MINI_MASK)
            
            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
                                                    gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(
                    image.shape, random_rois, gt_class_ids, gt_boxes)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask, mrcnn_part =\
                        build_detection_targets(
                            rpn_rois, gt_class_ids, gt_boxes, gt_masks, gt_parts, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)

                batch_key1s = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_key2s = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_key3s = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_identity_inds = np.zeros((batch_size, 1), dtype=np.int32)

                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                if config.USE_MINI_MASK:
                    batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                               config.MAX_GT_INSTANCES))
                else:
                    batch_gt_masks = np.zeros(
                        (batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
                batch_gt_parts = np.zeros((batch_size, image.shape[0], image.shape[1]), dtype=np.uint8)
                if random_rois:
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)
                        batch_mrcnn_part = np.zeros(
                            (batch_size,) + mrcnn_part.shape, dtype=mrcnn_part.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            batch_key1s[b] = mold_image(key1.astype(np.float32), config)
            batch_key2s[b] = mold_image(key2.astype(np.float32), config)
            batch_key3s[b] = mold_image(key3.astype(np.float32), config)
            batch_identity_inds[b, :] = identity_ind
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            batch_gt_parts[b, :, :] = gt_parts
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
                    batch_mrcnn_part[b] = mrcnn_part
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_key1s, batch_key2s, batch_key3s, batch_identity_inds,
                          batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks, batch_gt_parts]
                outputs = []

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend(
                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask, batch_mrcnn_part])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


def antipad(num=1):
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        return x[:, num:-num, num:-num, :]
    return KL.Lambda(func)
def flow_refinement_block_share(name_base, low_level_features, hight_level_features, stage, deconv_filters):
    assert len(low_level_features) == len(hight_level_features), "length of low_level_features and hight_level_features must be equal"
    deconv = KL.Conv2DTranspose(deconv_filters, (4, 4), strides=(2, 2), 
                           name='%sdeconv%d'%(name_base, stage-1))
    conv1 = KL.Conv2D(2, (3, 3), padding='same', name='%spredict_conv%d'%(name_base,stage))
    deconv1 = KL.Conv2DTranspose(2, (4, 4), strides=(2, 2), 
                                     name='%supsample_flow%dto%d'%(name_base, stage, stage-1))
    features = []
    for i in range(len(low_level_features)):
        #hight_level_feature deconv
        x = deconv(hight_level_features[i])
        x = antipad()(x)
        hight_deconv = KL.LeakyReLU(alpha=0.1)(x)

        #hight_level_feature predict flow
        flow = conv1(hight_level_features[i])
        #flow deconv
        flow_deconv = deconv1(flow)
        flow_deconv = antipad()(flow_deconv)

        res = KL.Concatenate(axis=-1)([low_level_features[i], hight_deconv, flow_deconv])
        features.append(res)
    return features

def flow_conv_block_share(name_base, input_tensors, filters, stage, kernel_size=3, padding=(1,1)):
    conv1 = KL.Conv2D(filters, (kernel_size, kernel_size), strides=(2, 2), name='%sconv%d'%(name_base,stage))
    conv2 = KL.Conv2D(filters, (3, 3), padding='same', name='%sconv%d_1'%(name_base,stage))

    features = []
    for i in range(len(input_tensors)):
        x = KL.ZeroPadding2D(padding=padding)(input_tensors[i])
        x = conv1(x)
        x = KL.LeakyReLU(alpha=0.1)(x)

        x = conv2(x)
        x = KL.LeakyReLU(alpha=0.1)(x)
        features.append(x)
    return features

def flownet2_S_with_scale_share(stage, input_a, input_b):
    assert len(input_a) == len(input_b), "length of input_a and input_b must be equal"

    num = len(input_a)
    name_base = 'flownet_s%d_'%stage

    conv1_layer = KL.Conv2D(64, (7, 7), strides=(2, 2), name=name_base + 'conv1')
    conv2_layer = KL.Conv2D(128, (5, 5), strides=(2, 2), name=name_base + 'conv2')
    pred_conv_layer = KL.Conv2D(2, (3, 3), padding='same', name=name_base+'predict_conv2')
    pred_scale_layer = KL.Conv2D(256, (1, 1), padding='same', name=name_base+'predict_scale',
        use_bias=False, kernel_initializer='zeros')

    conv2_features = []
    for i in range(num):
        input_tensor = KL.Concatenate(axis=-1)([input_a[i], input_b[i]])
        # conv1
        x = KL.ZeroPadding2D((3, 3))(input_tensor)
        x = conv1_layer(x)
        conv1 = KL.LeakyReLU(alpha=0.1)(x)

        # conv2
        x = KL.ZeroPadding2D((2, 2))(conv1)
        x = conv2_layer(x)
        conv2 = KL.LeakyReLU(alpha=0.1)(x)
        conv2_features.append(conv2)

    #conv3
    conv3_1_features = flow_conv_block_share(name_base, conv2_features, 256, 3, kernel_size=5, padding=(2,2))
    #conv4
    conv4_1_features = flow_conv_block_share(name_base, conv3_1_features, 512, 4)
    #conv5
    conv5_1_features = flow_conv_block_share(name_base, conv4_1_features, 512, 5)
    #conv6
    conv6_1_features = flow_conv_block_share(name_base, conv5_1_features, 1024, 6)

    concat5_features = flow_refinement_block_share(name_base, conv5_1_features, conv6_1_features, stage=6, deconv_filters=512)
    concat4_features = flow_refinement_block_share(name_base, conv4_1_features, concat5_features, stage=5, deconv_filters=256)
    concat3_features = flow_refinement_block_share(name_base, conv3_1_features, concat4_features, stage=4, deconv_filters=128)
    concat2_features = flow_refinement_block_share(name_base, conv2_features, concat3_features, stage=3, deconv_filters=64)

    flows = []
    scales = []
    for i in range(num):
        flow = pred_conv_layer(concat2_features[i])
        flows.append(flow)
        scale = pred_scale_layer(concat2_features[i])
        scale = KL.Lambda(lambda x: tf.add(x, 1))(scale)
        scales.append(scale)
    return flows, scales

def flow_image_preprocess_graph(image):
    # rescale to [-1, 1]
    image = image / 255.0
    # convert RGB --> BGR
    image = image[..., ::-1]
    # downsample the image to 1/2
    # shape = tf.shape(image)[1:3]
    # shape = shape / 2
    # shape = tf.cast(shape, tf.int32)
    # image = tf.image.resize_bilinear(image, shape)
    return image

def flow_postprocess_graph(flow, height, width):
    flow = flow * 20.0
    # TODO: Look at Accum (train) or Resample (deploy) to see if we need to do something different
    # flow = tf.image.resize_bilinear(flow,
    #                                 tf.stack([height, width]),
    #                                 align_corners=True)
    return flow

def conv_gru_unit(temporal_features, initial_state=None):
    input_tensor = KL.Lambda(lambda x: tf.stack(x, axis=1))(temporal_features)
    x = KL.ConvGRU2D(filters=256, kernel_size=(3, 3), name="gru_recurrent_unit", 
                   padding='same', return_sequences=False)(input_tensor, initial_state=initial_state)
    return x
def conv_lstm_unit(temporal_features, initial_state=None):
    input_tensor = KL.Lambda(lambda x: tf.stack(x, axis=1))(temporal_features)
    x = KL.ConvLSTM2D(filters=256, kernel_size=(3, 3), name="lstm_recurrent_unit", 
                   padding='same', return_sequences=False)(input_tensor, initial_state=initial_state)
    return x
def arbitrary_size_pooling(feature_map):
    b1 = tf.reduce_mean(feature_map, axis = 1, keep_dims=True)
    b2 = tf.reduce_mean(b1, axis = 2, keep_dims=True)
    return b2

def global_parsing_encoder_share(feature_map):
    conv1 = KL.Conv2D(256, (1, 1), padding='same', activation='relu', 
                      name='mrcnn_global_parsing_encoder_c1')
    # x1 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn0'%num_classes)(x1)

    conv2 = KL.Conv2D(256, (3, 3), padding='same', dilation_rate=(6, 6), activation='relu', 
                      name='mrcnn_global_parsing_encoder_c2')
    # x2 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn1'%num_classes)(x2)

    conv3 = KL.Conv2D(256, (3, 3), padding='same', dilation_rate=(12, 12), activation='relu', 
                      name='mrcnn_global_parsing_encoder_c3')
    # x3 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn2'%num_classes)(x3)

    conv4 = KL.Conv2D(256, (3, 3), padding='same', dilation_rate=(18, 18), activation='relu', 
                      name='mrcnn_global_parsing_encoder_c4')
    # x4 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn3'%num_classes)(x4)

    conv5 = KL.Conv2D(256, (1, 1), padding='same', activation='relu', 
                      name='mrcnn_global_parsing_encoder_c0')
    # x0 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn4'%num_classes)(x0)
    conv6 = KL.Conv2D(256, (1, 1), padding='same', activation='relu', 
                      name='mrcnn_global_parsing_encoder_conconv')


    features = []
    for i in range(len(feature_map)):
        
        x1 = conv1(feature_map[i])
        # x1 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn0'%num_classes)(x1)

        x2 = conv2(feature_map[i])
        # x2 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn1'%num_classes)(x2)

        x3 = conv3(feature_map[i])
        # x3 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn2'%num_classes)(x3)

        x4 = conv4(feature_map[i])
        # x4 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn3'%num_classes)(x4)

        x0 = KL.Lambda(lambda x: arbitrary_size_pooling(x))(feature_map[i])
        x0 = conv5(x0)
        # x0 = BatchNorm(axis=-1, name='mrcnn_global_parsing_%d_bn4'%num_classes)(x0)
        x0 = KL.Lambda(lambda x: tf.image.resize_bilinear(
                  x[0], tf.shape(x[1])[1:3], align_corners=True))([x0, feature_map[i]])


        x = KL.Lambda(lambda x: tf.concat(x, axis=-1))([x0, x1, x2, x3, x4])
        x = conv6(x)
        features.append(x)

    return features

def global_parsing_decoder_share(feature_map, low_feature_map):
    assert len(feature_map) == len(low_feature_map)
    conv1 = KL.Conv2D(48, (1, 1), padding='same', activation='relu', 
              name='mrcnn_global_parsing_decoder_conv1')
    conv2 = KL.Conv2D(256, (3, 3), padding='same', activation='relu', 
              name='mrcnn_global_parsing_decoder_conv2')
    conv3 = KL.Conv2D(256, (3, 3), padding='same', activation='relu', 
              name='mrcnn_global_parsing_decoder_conv3')
    features = []
    for i in range(len(feature_map)):

        # navie upsample from 1/16(32) to 1/4(128), fit the low_feature_map
        top = KL.Lambda(lambda x: tf.image.resize_bilinear(
                  x[0], tf.shape(x[1])[1:3], align_corners=True))([feature_map[i], low_feature_map[i]])
        # low dim of low_feature_map by 1*1 conv
        low = conv1(low_feature_map[i])

        # x = KL.Concatenate(axis=-1)([top, low])
        x = KL.Lambda(lambda x: tf.concat(x, axis=-1))([top, low])
        x = conv2(x)
        x = conv3(x)
        features.append(x)
    return features

def global_parsing_share(feature_map, num_classes):
    conv1 = KL.Conv2D(num_classes, (3, 3), padding='same', dilation_rate=(6, 6), 
                      name='mrcnn_global_parsing_c1')
    conv2 = KL.Conv2D(num_classes, (3, 3), padding='same', dilation_rate=(12, 12), 
                      name='mrcnn_global_parsing_c2')
    conv3 = KL.Conv2D(num_classes, (3, 3), padding='same', dilation_rate=(18, 18), 
                      name='mrcnn_global_parsing_c3')

    features = []
    for i in range(len(feature_map)):

        x1 = conv1(feature_map[i])
        x2 = conv2(feature_map[i])
        x3 = conv3(feature_map[i])
        x = KL.Add()([x1, x2 ,x3])
        features.append(x)
    return features

def get_option_inds_graph(input_key_identity):
    """
    input_key_identity: [batch_size, 1]
    """
    b = tf.shape(input_key_identity)[0]
    batch_inds = tf.range(tf.constant(0, tf.int32), b)
    batch_inds = tf.expand_dims(batch_inds, axis=-1)
    option_inds = tf.concat([batch_inds, tf.cast(input_key_identity, tf.int32)], axis=-1)
    return option_inds


class ATEN_PARSING_RCNN():
    """Encapsulates the Mask RCNN model functionality.

    The actual Keras model is in the keras_model property.
    """

    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        """Build Mask R-CNN architecture.
            input_shape: The shape of the input image.
            mode: Either "training" or "inference". The inputs and
                outputs of the model differ accordingly.
        """
        assert mode in ['training', 'inference']

        # Image size must be dividable by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling.")

        # Inputs
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_image_key1 = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image_key1")
        input_image_key2 = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image_key2")
        input_image_key3 = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image_key3")
        input_key_identity = KL.Input(
            batch_shape=[config.BATCH_SIZE, 1],
            name="input_key_identity", dtype=tf.int32)
        input_image_meta = KL.Input(shape=[None], name="input_image_meta")
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)

            # Detection GT (class IDs, bounding boxes, and masks)
            # 1. GT Class IDs (zero padded)
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)
            # 2. GT Boxes in pixels (zero padded)
            # [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)] in image coordinates
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)
            # Normalize coordinates
            h, w = K.shape(input_image)[1], K.shape(input_image)[2]
            image_scale = K.cast(K.stack([h, w, h, w], axis=0), tf.float32)
            gt_boxes = KL.Lambda(lambda x: x / image_scale)(input_gt_boxes)
            # 3. GT Masks (zero padded)
            # [batch, height, width, MAX_GT_INSTANCES]
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(
                    shape=[config.MINI_MASK_SHAPE[0],
                           config.MINI_MASK_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                    name="input_gt_masks", dtype=bool)
            # 4. GT Part
            input_gt_part = KL.Input(
                    shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1]],
                    name="input_gt_part", dtype=tf.uint8)

        # Build the shared convolutional layers.
        # Bottom-up Layers
        # Returns a list of the last layers of each stage, 5 in total.
        # Don't create the thead (stage 5), so we pick the 4th item in the list.
        c1_features, c5_features = deeplab_resnet_share([input_image_key1, 
            input_image_key2, input_image_key3], 'resnet50')
        coarse_features = global_parsing_encoder_share(c5_features)
        fine_features = global_parsing_decoder_share(coarse_features, c1_features)

        # ------------ATEN-----------------------------------------------
        feature_map_key1 = fine_features[0]
        feature_map_key2 = fine_features[1]
        feature_map_key3 = fine_features[2]
        #-----------flownet-------------------------------
        # image preprocess
        flownet_input_key3  = KL.Lambda(lambda x: flow_image_preprocess_graph(x))(input_image_key3)
        flownet_input_key2  = KL.Lambda(lambda x: flow_image_preprocess_graph(x))(input_image_key2)
        flownet_input_key1  = KL.Lambda(lambda x: flow_image_preprocess_graph(x))(input_image_key1)
        flownet_input_cur  = KL.Lambda(lambda x: flow_image_preprocess_graph(x))(input_image)
        # flownet_S
        flows, scales = flownet2_S_with_scale_share(1, 
            input_a=[flownet_input_cur, flownet_input_key1, flownet_input_key1], 
            input_b=[flownet_input_key1, flownet_input_key2, flownet_input_key3])

        new_2_cur_raw_flow = flows[0]
        new_2_cur_scale = scales[0]
        key2_2_key1_raw_flow = flows[1]
        key2_2_key1_scale = scales[1]
        key3_2_key1_raw_flow = flows[2]
        key3_2_key1_scale = scales[2]

        new_2_cur_feature_flow = KL.Lambda(lambda x: flow_postprocess_graph(x, 
            config.BACKBONE_SHAPES[0][0], 
            config.BACKBONE_SHAPES[0][1]))(new_2_cur_raw_flow)
        key2_2_key1_feature_flow = KL.Lambda(lambda x: flow_postprocess_graph(x, 
            config.BACKBONE_SHAPES[0][0], 
            config.BACKBONE_SHAPES[0][1]))(key2_2_key1_raw_flow)
        key3_2_key1_feature_flow = KL.Lambda(lambda x: flow_postprocess_graph(x, 
            config.BACKBONE_SHAPES[0][0], 
            config.BACKBONE_SHAPES[0][1]))(key3_2_key1_raw_flow)

        feature_map_key2 = KL.Lambda(lambda x: flow_warp(x[0], x[1]))([feature_map_key2, 
            key2_2_key1_feature_flow])
        feature_map_key2 = KL.Lambda(lambda x: tf.multiply(x[0], x[1]))([feature_map_key2, key2_2_key1_scale])
        feature_map_key3 = KL.Lambda(lambda x: flow_warp(x[0], x[1]))([feature_map_key3, 
            key3_2_key1_feature_flow])
        feature_map_key3 = KL.Lambda(lambda x: tf.multiply(x[0], x[1]))([feature_map_key3, key3_2_key1_scale])
        #--------------------------------------------------
        init_feature_map = KL.Lambda(lambda x: (x[0] + x[1] + x[2])/3)([feature_map_key3, 
            feature_map_key2, feature_map_key1])
        if config.RECURRENT_UNIT == 'lstm':
            feature_map_key1 = conv_lstm_unit([feature_map_key3, feature_map_key2, feature_map_key1], 
                initial_state=None)
        elif config.RECURRENT_UNIT == 'gru':
            feature_map_key1 = conv_gru_unit([feature_map_key3, feature_map_key2, feature_map_key1], 
                initial_state=init_feature_map)
        # warping the aggregated_key_feature to cur_feature by optical flow
        feature_map_cur = KL.Lambda(lambda x: flow_warp(x[0], x[1]))([feature_map_key1, 
            new_2_cur_feature_flow])
        feature_map_cur = KL.Lambda(lambda x: tf.multiply(x[0], x[1]))([feature_map_cur, new_2_cur_scale])

        # recognize wether key frame
        option_feature_map = KL.Lambda(lambda x: tf.stack(x, axis=1))([feature_map_key1, feature_map_cur])
        choose_ind = KL.Lambda(lambda x: get_option_inds_graph(x))(input_key_identity)
        final_feature = KL.Lambda(lambda x:tf.gather_nd(x[0], tf.cast(x[1], tf.int32)))([option_feature_map, choose_ind])

        # --------task specific sub-network------------------------------
        # input final_feature
        # global parsing branch
        global_parsing_map = global_parsing_share([final_feature], config.NUM_PART_CLASS)[0]

        rpn_feature_map = KL.Conv2D(256, (3, 3), activation='relu', padding='same', 
            name='mrcnn_share_rpn_conv1')(final_feature)
        rpn_feature_map = KL.Conv2D(256, (3, 3), activation='relu', padding='same', 
            name='mrcnn_share_rpn_conv2')(rpn_feature_map)

        mrcnn_feature_map = KL.Conv2D(256, (3, 3), activation='relu', padding='same', 
            name='mrcnn_share_recog_conv1')(final_feature)
        mrcnn_feature_map = KL.Conv2D(256, (3, 3), activation='relu', padding='same', 
            name='mrcnn_share_recog_conv2')(mrcnn_feature_map)

        # Generate Anchors
        self.anchors = utils.generate_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             config.BACKBONE_SHAPES[0],
                                             config.BACKBONE_STRIDES[0],
                                             config.RPN_ANCHOR_STRIDE)

        # RPN Model
        rpn_class_logits, rpn_class, rpn_bbox = rpn_graph(rpn_feature_map, 
                                                    len(config.RPN_ANCHOR_RATIOS) * len(config.RPN_ANCHOR_SCALES),
                                                    config.RPN_ANCHOR_STRIDE)

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training"\
            else config.POST_NMS_ROIS_INFERENCE
        pre_proposal_count = config.PRE_NMS_ROIS_TRAINING if mode == "training"\
            else config.PRE_NMS_ROIS_INFERENCE
        rpn_rois = ProposalLayer(proposal_count=proposal_count,
                                 pre_proposal_count=pre_proposal_count,
                                 nms_threshold=config.RPN_NMS_THRESHOLD,
                                 name="ROI",
                                 anchors=self.anchors,
                                 config=config)([rpn_class, rpn_bbox])

        if mode == "training":
            # Class ID mask to mark class IDs supported by the dataset the image
            # came from.
            _, _, _, active_class_ids = KL.Lambda(lambda x: parse_image_meta_graph(x),
                                                  mask=[None, None, None, None])(input_image_meta)

            if not config.USE_RPN_ROIS:
                raise Exception("not support use another roi except rpn roi")

            target_rois = rpn_rois

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask =\
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            # TODO: verify that this handles zero padded ROIs
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rois, mrcnn_feature_map, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_CLASSES)

            mrcnn_mask = build_fpn_mask_graph(rois, mrcnn_feature_map,
                                              config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES)

            # TODO: clean up (use tf.identify if necessary)
            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)

            global_parsing_loss = KL.Lambda(lambda x: mrcnn_global_parsing_loss_graph(config.NUM_PART_CLASS, *x), name="mrcnn_global_parsing_loss")(
                [input_gt_part, global_parsing_map])


            # Losses
            rpn_class_loss = KL.Lambda(lambda x: rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name="mrcnn_class_loss")(
                [target_class_ids, mrcnn_class_logits, active_class_ids])
            bbox_loss = KL.Lambda(lambda x: mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])
            mask_loss = KL.Lambda(lambda x: mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                [target_mask, target_class_ids, mrcnn_mask])

            # Model
            inputs = [input_image, input_image_key1, input_image_key2, input_image_key3, 
                      input_key_identity, input_image_meta, input_rpn_match, input_rpn_bbox, 
                      input_gt_class_ids, input_gt_boxes, input_gt_masks, input_gt_part]
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_mask,
                       rpn_rois, output_rois,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss, mask_loss, 
                       global_parsing_loss]
            model = KM.Model(inputs, outputs, name='aten')
        else:
            # Network Heads
            # Proposal classifier and BBox regressor heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox =\
                fpn_classifier_graph(rpn_rois, mrcnn_feature_map, config.IMAGE_SHAPE,
                                     config.POOL_SIZE, config.NUM_CLASSES)

            # Detections
            # output is [batch, num_detections, (y1, x1, y2, x2, class_id, score)] in image coordinates
            detections = DetectionLayer(config, name="mrcnn_detection")(
                [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])

            # Convert boxes to normalized coordinates
            # TODO: let DetectionLayer return normalized coordinates to avoid
            #       unnecessary conversions
            h, w = config.IMAGE_SHAPE[:2]
            detection_boxes = KL.Lambda(
                lambda x: x[..., :4] / np.array([h, w, h, w]))(detections)

            # Create masks for detections
            mrcnn_mask = build_fpn_mask_graph(detection_boxes, mrcnn_feature_map,
                                              config.IMAGE_SHAPE,
                                              config.MASK_POOL_SIZE,
                                              config.NUM_CLASSES)

            global_parsing_prob = KL.Lambda(lambda x: post_processing_graph(*x))([global_parsing_map, input_image])

            model = KM.Model([input_image, input_image_key1, input_image_key2, input_image_key3, 
                                 input_key_identity, input_image_meta],
                             [detections, mrcnn_class, mrcnn_bbox,
                                 mrcnn_mask, rpn_rois, rpn_class, rpn_bbox, global_parsing_prob],
                             name='aten')

        # Add multi-GPU support.
        if config.GPU_COUNT > 1:
            from parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)

        return model

    def find_last(self):
        """Finds the last checkpoint file of the last trained model in the
        model directory.
        Returns:
            log_dir: The directory where events and weights are saved
            checkpoint_path: the path to the last checkpoint file
        """
        # Get directory names. Each directory corresponds to a model
        dir_names = next(os.walk(self.model_dir))[1]
        key = self.config.NAME.lower()
        dir_names = filter(lambda f: f.startswith(key), dir_names)
        dir_names = sorted(dir_names)
        if not dir_names:
            return None, None
        # Pick last directory
        dir_name = os.path.join(self.model_dir, dir_names[-1])
        # Find the last checkpoint
        checkpoints = next(os.walk(dir_name))[2]
        checkpoints = filter(lambda f: f.startswith("aten"), checkpoints)
        checkpoints = sorted(checkpoints)
        if not checkpoints:
            return dir_name, None
        checkpoint = os.path.join(dir_name, checkpoints[-1])
        return dir_name, checkpoint

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the correspoding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exlude: list of layer names to excluce
        """
        import h5py
        from keras.engine import topology

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        # # exclude some layers
        if exclude:
            layers = filter(lambda l: exclude.match(l.name)==None, layers)

        # layers_name = [l.name for l in layers]
        print("load model", filepath)

        if by_name:
            topology.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            topology.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def save_weights(self, filepath):
        keras_model = self.keras_model
        model = keras_model.inner_model if hasattr(keras_model, "inner_model")\
            else keras_model
        model.save_weights(filepath)


    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(lr=learning_rate, momentum=momentum,
                                         clipnorm=5.0)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = ["rpn_class_loss", "rpn_bbox_loss",
                      "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss", 
                      "mrcnn_global_parsing_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            self.keras_model.add_loss(
                tf.reduce_mean(layer.output, keep_dims=True))

        # Add L2 Regularization
        reg_losses = [keras.regularizers.l2(self.config.WEIGHT_DECAY)(w)
                      for w in self.keras_model.trainable_weights
                      if ('gamma' not in w.name) and ('beta' not in w.name) and ('bias' not in w.name)]

        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(optimizer=optimizer, loss=[
                                 None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            self.keras_model.metrics_tensors.append(tf.reduce_mean(
                layer.output, keep_dims=True))


    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=1):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model")\
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainble layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5
            regex = r".*/\w+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})/aten\_\w+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                                        int(m.group(4)), int(m.group(5)))
                self.epoch = int(m.group(6)) + 1

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(
            self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "aten_{}_*epoch*.h5".format(
            self.config.NAME.lower()))
        self.checkpoint_path = self.checkpoint_path.replace(
            "*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, layers, period):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
        """
        assert self.mode == "training", "Create model in training mode."
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "mask_heads": r"(mrcnn\_bbox\_.*)|(rpn\_.*)|(mrcnn\_class\_.*)|(mrcnn\_mask\_.*)|(mrcnn\_share\_.*)",
            "heads": r"(mrcnn\_.*)|(rpn\_.*)",
            # From a specific Resnet stage and up
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(mrcnn\_.*)|(rpn\_.*)",
            # All layers
            "all": ".*",
            "first_stage": r"(.*\_recurrent\_unit)|(mrcnn\_bbox\_.*)|(rpn\_.*)|(mrcnn\_class\_.*)|(mrcnn\_mask\_.*)|(mrcnn\_share\_.*)|(mrcnn\_global\_parsing\_c.*)",
            "second_stage": r"(.*\_recurrent\_unit)|(mrcnn\_.*)|(rpn\_.*)|(flownet\_.*)",
        }
        if layers in layer_regex.keys():
            layers = layer_regex[layers]

        # Data generators
        train_generator = data_generator(train_dataset, self.config, shuffle=True, 
                                         batch_size=self.config.BATCH_SIZE)
        val_generator = data_generator(val_dataset, self.config, shuffle=True,
                                       batch_size=self.config.BATCH_SIZE)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, period=period,
                                            verbose=0, save_weights_only=True),
        ]

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=max(self.config.BATCH_SIZE // 2, 2),
            use_multiprocessing=True,
            verbose=1,
        )
        self.epoch = max(self.epoch, epochs)





    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def mold_inputs(self, images):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matricies [height,width,depth]. Images can have
            different sizes.

        Returns 3 Numpy matricies:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        molded_images = []
        image_metas = []
        windows = []
        for image in images:
            # Resize image to fit the model expected size
            # TODO: move resizing to mold_image()
            molded_image, window, scale, padding = utils.resize_image(
                image,
                min_dim=self.config.IMAGE_MIN_DIM,
                max_dim=self.config.IMAGE_MAX_DIM,
                padding=self.config.IMAGE_PADDING)
            molded_image = mold_image(molded_image, self.config)
            # Build image_meta
            image_meta = compose_image_meta(
                0, image.shape, window,
                np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
            # Append
            molded_images.append(molded_image)
            windows.append(window)
            image_metas.append(image_meta)
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def unmold_detections(self, detections, mrcnn_mask, mrcnn_global_parsing, image_shape, window):
        """Reformats the detections of one image from the format of the neural
        network output to a format suitable for use in the rest of the
        application.

        detections: [N, (y1, x1, y2, x2, class_id, score)]
        mrcnn_mask: [N, height, width, num_classes]
        mrcnn_global_parsing: [resized_height, resized_width, num_classes]
        image_shape: [height, width, depth] Original size of the image before resizing
        window: [y1, x1, y2, x2] Box in the image where the real image is
                excluding the padding.

        Returns:
        boxes: [N, (y1, x1, y2, x2)] Bounding boxes in pixels
        class_ids: [N] Integer class IDs for each bounding box
        scores: [N] Float probability scores of the class_id
        masks: [height, width, num_instances] Instance masks
        """
        # How many detections do we have?
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]

        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Compute scale and shift to translate coordinates to image domain.
        h_scale = image_shape[0] / (window[2] - window[0])
        w_scale = image_shape[1] / (window[3] - window[1])
        scale = min(h_scale, w_scale)
        shift = window[:2]  # y, x
        scales = np.array([scale, scale, scale, scale])
        shifts = np.array([shift[0], shift[1], shift[0], shift[1]])

        # Translate bounding boxes to image domain
        boxes = np.multiply(boxes - shifts, scales).astype(np.int32)

        # Filter out detections with zero area. Often only happens in early
        # stages of training when the network weights are still a bit random.
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]

        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        for i in range(N):
            # Convert neural network mask to full size mask
            full_mask = utils.unmold_mask(masks[i], boxes[i], image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty((0,) + masks.shape[1:3])

        global_parsing = mrcnn_global_parsing[window[0]:window[2], window[1]:window[3], :]
        global_parsing = skimage.transform.resize(global_parsing, (image_shape[0], image_shape[1]), mode="constant")

        return boxes, class_ids, scores, full_masks, global_parsing

    def detect(self, images, key1s, key2s, key3s, identity_ind):
        """Runs the detection pipeline.

        images: List of images, potentially of different sizes.

        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes
        class_ids: [N] int class IDs
        scores: [N] float probability scores for the class IDs
        masks: [H, W, N] instance binary masks
        """
        assert len(images) == len(identity_ind
            ) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"

        # Mold inputs to format expected by the neural network
        molded_images, image_metas, windows = self.mold_inputs(images)
        molded_key1s, _, _ = self.mold_inputs(key1s)
        molded_key2s, _, _ = self.mold_inputs(key2s)
        molded_key3s, _, _ = self.mold_inputs(key3s)
        identity_ind = np.stack(identity_ind)

        # Run object detection
        detections, mrcnn_class, mrcnn_bbox, mrcnn_mask, \
            rois, rpn_class, rpn_bbox, mrcnn_global_parsing_prob =\
            self.keras_model.predict([molded_images, molded_key1s, molded_key2s, molded_key3s, 
                identity_ind, image_metas], verbose=0)
        # Process detections
        results = []
        for i, image in enumerate(images):
            final_rois, final_class_ids, final_scores, final_masks, final_globals =\
                self.unmold_detections(detections[i], mrcnn_mask[i], mrcnn_global_parsing_prob[i], 
                                       image.shape, windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "global_parsing": final_globals
            })
        return results
