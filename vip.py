"""
Written by Qixian Zhou
"""

import math
import random
import os
import numpy as np
import cv2
import scipy
import scipy.io as sio
from config import Config
import skimage.io
import skimage.transform
import utils


class ParsingRCNNModelConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "parsing_rcnn_model_config"

    SAVE_MODEL_PERIOD = 5

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 4
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # background + person

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 450
    IMAGE_MAX_DIM = 512

    # If True, pad images with zeros such that they're (max_dim by max_dim)
    IMAGE_PADDING = True  # currently, the False option is not supported

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    STEPS_PER_EPOCH = 1000

    VALIDATION_STEPS = 30

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128
    # Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a deeplab backbone.
    BACKBONE_STRIDES = [4, 8, 16, 16, 32]


    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256
    # Non-max suppression threshold to filter RPN proposals.
    # You can reduce this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 384)
    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 0.75, 1]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1


    # ROIs kept after non-maximum supression (training and inference)
    PRE_NMS_ROIS_TRAINING = 12000
    PRE_NMS_ROIS_INFERENCE = 6000
    POST_NMS_ROIS_TRAINING = 2000
    POST_NMS_ROIS_INFERENCE = 1000

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 100

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.7

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 100

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimzer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Use RPN ROIs or externally generated ROIs for training
    # Keep this True for most situations. Set to False if you want to train
    # the head branches on ROI generated by code rather than the ROIs from
    # the RPN. For example, to debug the classifier head without having to
    # train the RPN.
    USE_RPN_ROIS = True

    # parsing part class num
    NUM_PART_CLASS = 1 + 19 #background + classes



class VideoModelConfig(ParsingRCNNModelConfig):
    NAME = 'video_aten_model'
    KEY_RANGE_L = 3
    RECURRENT_UNIT = "gru"
    assert RECURRENT_UNIT in ["gru", "lstm"]

class VIPDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """
    def add_parsing_class(self, source, class_id, class_name):
        # Does the class exist already?
        for info in self.parsing_class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.parsing_class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })


    def load_vip(self, dataset_dir, subset):
        # Add classes
        self.add_class("VIP", 1, "person")
        self.add_parsing_class("VIP", 1, "hat")
        self.add_parsing_class("VIP", 2, "hair")
        self.add_parsing_class("VIP", 3, "gloves")
        self.add_parsing_class("VIP", 4, "sun-glasses")
        self.add_parsing_class("VIP", 5, "upper-clothes")
        self.add_parsing_class("VIP", 6, "dress")
        self.add_parsing_class("VIP", 7, "coat")
        self.add_parsing_class("VIP", 8, "socks")
        self.add_parsing_class("VIP", 9, "pants")
        self.add_parsing_class("VIP", 10, "torso-skin")
        self.add_parsing_class("VIP", 11, "scarf")
        self.add_parsing_class("VIP", 12, "skirt")
        self.add_parsing_class("VIP", 13, "face")
        self.add_parsing_class("VIP", 14, "left-arm")
        self.add_parsing_class("VIP", 15, "right-arm")
        self.add_parsing_class("VIP", 16, "left-leg")
        self.add_parsing_class("VIP", 17, "right-leg")
        self.add_parsing_class("VIP", 18, "left-shoe")
        self.add_parsing_class("VIP", 19, "right-shoe")

        # Path
        image_dir = os.path.join(dataset_dir, 'Images')
        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        rfp = open(os.path.join(dataset_dir, 'lists', '%s_id.txt'%subset), 'r')
        for line in rfp.readlines():
            image_id = line.strip()
            self.add_image("VIP", image_id=image_id, 
                           path=os.path.join(image_dir, '%s.jpg'%image_id), 
                           front_frame_list=os.path.join(dataset_dir, 'front_frame_list', '%s.txt'%image_id),
                           behind_frame_list=os.path.join(dataset_dir, 'behind_frame_list', '%s.txt'%image_id),
                           inst_anno=os.path.join(dataset_dir, 'Human_ids', '%s.png'%image_id),
                           part_anno=os.path.join(dataset_dir, 'Category_ids', '%s.png'%image_id),
                           part_rev_anno=os.path.join(dataset_dir, 'Category_rev_ids', '%s.png'%image_id))

    def load_keys(self, image_id, key_range, key_num):
        image_info = self.image_info[image_id]
        with open(image_info['front_frame_list'], 'r') as fp:
            front_frame_ids = [x.strip() for x in fp.readlines()]
        with open(image_info['behind_frame_list'], 'r') as fp:
            behind_frame_ids = [x.strip() for x in fp.readlines()]

        # get image_dir
        image_dir = image_info['path']
        image_dir = image_dir[:image_dir.rfind('.')]
        adjacent_frames_dir = image_dir.replace("Images", "adjacent_frames")
        frame_ind = int(image_dir.split('/')[-1])

        if len(front_frame_ids)>0:
            most_left_ind = int(front_frame_ids[-1])
        else:
            most_left_ind = frame_ind
        if len(behind_frame_ids)>0:
            most_right_ind = int(behind_frame_ids[-1])
        else:
            most_right_ind = frame_ind
        #print(most_left_ind, frame_ind, most_right_ind)
        new_key_left = max(most_left_ind, frame_ind - int(key_range//2))
        new_key_right = min(most_right_ind, frame_ind + (key_range - 1- int(key_range//2) ) )
        sel_pool = []
        for i in range(new_key_left, new_key_right+1):
            if i != frame_ind:
                sel_pool.append([i, 1])
            else:
                sel_pool.append([i, 0])
        sel = random.choice(sel_pool)
        new_key_ind = sel[0]
        identity_ind = sel[1]
        #print("new key ind", new_key_ind)
        keys = []
        most_left_key_ind = new_key_ind - key_range * (key_num - 1)
        most_right_key_ind = new_key_ind + key_range * (key_num - 1)
        get_left = True
        #print(most_left_key_ind, most_right_key_ind)
        if random.randint(0, 1):
            # if <most_left_ind, get right keys
            if most_left_key_ind < most_left_ind:
                get_left = False
                if most_right_key_ind > most_right_ind:
                    raise Exception("There is not enough adjacent frames, right_key %d VS right_limit %d"%(most_right_key_ind, most_right_ind)) 
            else:
                get_left = True
        else:
            # if >most_right_ind, get left keys
            if most_right_key_ind > most_right_ind:
                get_left = True
                if most_left_key_ind < most_left_ind:
                    raise Exception("There is not enough adjacent frames left_key %d VS left_limit %d"%(most_left_key_ind, most_left_ind)) 
            else:
                get_left = False
        if get_left:
            if identity_ind == 0:
                tmp_path = image_info['path']
            else:
                tmp_path = '%s/%012d.jpg'%(adjacent_frames_dir, new_key_ind)
            tmp_key = skimage.io.imread(tmp_path)
            keys.append(tmp_key)
            for i in range(1, key_num):
                tmp_ind = new_key_ind - i * key_range
                tmp_path = '%s/%012d.jpg'%(adjacent_frames_dir, tmp_ind)
                tmp_key = skimage.io.imread(tmp_path)
                keys.append(tmp_key)
        else:
            if identity_ind == 0:
                tmp_path = image_info['path']
            else:
                tmp_path = '%s/%012d.jpg'%(adjacent_frames_dir, new_key_ind)
            tmp_key = skimage.io.imread(tmp_path)
            keys.append(tmp_key)
            for i in range(1, key_num):
                tmp_ind = new_key_ind + i * key_range
                tmp_path = '%s/%012d.jpg'%(adjacent_frames_dir, tmp_ind)
                tmp_key = skimage.io.imread(tmp_path)
                keys.append(tmp_key)

        return keys, identity_ind

    def load_infer_keys(self, image_id, key_range, key_num):
        image_info = self.image_info[image_id]
        with open(image_info['front_frame_list'], 'r') as fp:
            front_frame_ids = [x.strip() for x in fp.readlines()]
        with open(image_info['behind_frame_list'], 'r') as fp:
            behind_frame_ids = [x.strip() for x in fp.readlines()]

        # get image_dir
        image_dir = image_info['path']
        image_dir = image_dir[:image_dir.rfind('.')]
        adjacent_frames_dir = image_dir.replace("Images", "adjacent_frames")
        frame_ind = int(image_dir.split('/')[-1])

        if len(front_frame_ids)>0:
            most_left_ind = int(front_frame_ids[-1])
        else:
            most_left_ind = frame_ind
        if len(behind_frame_ids)>0:
            most_right_ind = int(behind_frame_ids[-1])
        else:
            most_right_ind = frame_ind
        #print(most_left_ind, frame_ind, most_right_ind)
        new_key_left = max(most_left_ind, frame_ind - int(key_range//2))
        new_key_right = min(most_right_ind, frame_ind + (key_range - 1- int(key_range//2) ) )
        sel_pool = []
        for i in range(new_key_left, new_key_right+1):
            if i != frame_ind:
                sel_pool.append([i, 1])
            else:
                sel_pool.append([i, 0])
        sel = random.choice(sel_pool)
        new_key_ind = sel[0]
        identity_ind = sel[1]
        keys = []
        most_left_key_ind = new_key_ind - key_range * (key_num - 1)
        most_right_key_ind = new_key_ind + key_range * (key_num - 1)
        get_left = True

        # if < most_left_ind, get right keys
        if most_left_key_ind < most_left_ind:
            get_left = False
            if most_right_key_ind > most_right_ind:
                raise Exception("There is not enough adjacent frames, right_key %d VS right_limit %d"%(most_right_key_ind, most_right_ind)) 
        else:
            get_left = True

        if get_left:
            if identity_ind == 0:
                tmp_path = image_info['path']
            else:
                tmp_path = '%s/%012d.jpg'%(adjacent_frames_dir, new_key_ind)
            tmp_key = skimage.io.imread(tmp_path)
            keys.append(tmp_key)
            for i in range(1, key_num):
                tmp_ind = new_key_ind - i * key_range
                tmp_path = '%s/%012d.jpg'%(adjacent_frames_dir, tmp_ind)
                tmp_key = skimage.io.imread(tmp_path)
                keys.append(tmp_key)
        else:
            if identity_ind == 0:
                tmp_path = image_info['path']
            else:
                tmp_path = '%s/%012d.jpg'%(adjacent_frames_dir, new_key_ind)
            tmp_key = skimage.io.imread(tmp_path)
            keys.append(tmp_key)
            for i in range(1, key_num):
                tmp_ind = new_key_ind + i * key_range
                tmp_path = '%s/%012d.jpg'%(adjacent_frames_dir, tmp_ind)
                tmp_key = skimage.io.imread(tmp_path)
                keys.append(tmp_key)

        return keys, identity_ind


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.image_info[image_id]
        if image_info["source"] != "VIP":
            return super(VIPDataset, self).load_mask(image_id)

        class_ids = []
        instance_masks = []

        gt_inst_data = cv2.imread(image_info["inst_anno"], cv2.IMREAD_GRAYSCALE)
        # get all instance label list
        unique_inst = np.unique(gt_inst_data)
        background_ind = np.where(unique_inst == 0)[0]
        unique_inst = np.delete(unique_inst, background_ind)

        # print("unique_inst", unique_inst)

        for i in range(unique_inst.shape[0]):
            inst_mask = unique_inst[i]
            # get specific instance mask
            im_mask = (gt_inst_data == inst_mask)
            instance_masks.append(im_mask)
            class_ids.append(1)
        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(VIPDataset, self).load_mask(image_id)

    def load_part(self, image_id):
        """Load part category map for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a category_id map [height, width].

        Returns:
        parts: A uint8 array of shape [height, width].
        """
        image_info = self.image_info[image_id]
        gt_part_data = cv2.imread(image_info["part_anno"], cv2.IMREAD_GRAYSCALE)
        return gt_part_data.astype(np.uint8)

    def load_reverse_part(self, image_id):
        """Load part category map for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a category_id map [height, width].

        Returns:
        parts: A uint8 array of shape [height, width].
        """
        image_info = self.image_info[image_id]
        gt_part_data = cv2.imread(image_info["part_rev_anno"], cv2.IMREAD_GRAYSCALE)
        return gt_part_data.astype(np.uint8)

