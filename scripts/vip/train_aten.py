import os
import sys
sys.path.insert(0, os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
import numpy as np

from vip import VideoModelConfig
from vip import VIPDataset
import utils
import aten_model as modellib

class trainConfig(VideoModelConfig):
    NAME = "vip_video_"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    STEPS_PER_EPOCH = 3100
    VALIDATION_STEPS = 450
    SAVE_MODEL_PERIOD = 1
    # Weight decay regularization
    WEIGHT_DECAY = 0.0001
    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])
    KEY_RANGE_L = 3
    RECURRENT_UNIT = 'gru'

# Root directory of the project
ROOT_DIR = os.getcwd()
# Path to trained weights file
PRETRAIN_MODEL_PATH = os.path.join(ROOT_DIR, "models", "aten_p2l3.h5")
PARSING_RCNN_MODEL_PATH = os.path.join(ROOT_DIR, "models", "parsing_rcnn.h5")
FLOWNET_MODEL_PATH = os.path.join(ROOT_DIR, "models", "flownet2-S.h5")
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "./outputs"
DEFAULT_DATASET_DIR = "/share/HDD/zhouqixian/dataset/VIP"
############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN on Pascal Person Part.')
    parser.add_argument('--dataset', required=False,
                        default=DEFAULT_DATASET_DIR,
                        metavar="/path/to/coco/",
                        help='Directory of the dataset')
    parser.add_argument('--model', required=False,
                        default="pretrain",
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')

    args = parser.parse_args()
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    config = trainConfig()
    config.display()

    # Create model
    model = modellib.ATEN_PARSING_RCNN(mode="training", config=config,
                              model_dir=args.logs)
    # Select weights file to load
    if args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "pretrain":
        model_path = PRETRAIN_MODEL_PATH
    else:
        model_path = args.model

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path, by_name=True)

    # Training dataset. Use the training set and 35K from the
    # validation set, as as in the Mask RCNN paper.
    dataset_train = VIPDataset()
    dataset_train.load_vip(args.dataset, "trainval")
    dataset_train.prepare()
 
    # Validation dataset
    dataset_val = VIPDataset()
    dataset_val.load_vip(args.dataset, "test")
    dataset_val.prepare()
 
    # *** This training schedule is an example. Update to your needs ***

    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=0.001,
                epochs=70,
                layers='all',
                period=config.SAVE_MODEL_PERIOD)
