# Adaptive Temporal Encoding Network for Video Instance-level Human Parsing
By Qixian Zhou, Xiaodan Liang, Ke Gong, Liang Lin (ACM MM18)

## Requirements
Python3, TensorFlow 1.3+, Keras 2.0.8+

## Dataset
The model is trained and evaluated on our proposed [VIP dataset](http://sysu-hcp.net/lip/video_parsing.php) for video instance-level human parsing. Please check it for more model details.

## Models
Models are released on google drive and [baidu drive](https://pan.baidu.com/s/1tZfm3Prvzn47cZi5RZ-lNw):

Parsing-RCNN(frame-level) weights(aten_p2l3.h5).

ATEN(p=2,l=3) weights(parsing_rcnn.h5).

## Installation
1. Clone this repository
2. Keras with convGRU2D installation.
```Bash
cd keras_convGRU
python setup.py install
```
3. flow_warp ops compile(optional). The flow_warp.so have been generated(Ubuntu14.04, gcc4.8.4, python3.6, tf1.4). To compile flow_warp ops, you can excute the code as follows:
```Bash
cd ops
make
```
4. Dataset setup. Download the [VIP dataset](http://sysu-hcp.net/lip/video_parsing.php)(both VIP_Fine and VIP_Sequence) and decompress them. The directory structure of VIP should be as follows:
VIP
    Images
        videos1  
        ...  
        videos404  
    adjacent_frames  
        videos1  
        ...  
        videos404  
    behind_frame_list  
    front_frame_list  
    Categorys  
    Category_ids  
    Category_rev_ids  
    Human  
    Human_ids  
    Instances  
    Instance_ids  
    lists  

5. Model setup
download released weights and place in models floder.

## Training
```Bash
# ATEN training on VIP
python scripts/vip/train_aten.py

# Parsing-RCNN(frame-level) training on VIP
python scripts/vip/train_parsingrcnn.py
```

## Inference
```Bash
# ATEN inference on VIP
python scripts/vip/test_aten.py

# Parsing-RCNN(frame-level) inference on VIP
python scripts/vip/test_parsingrcnn.py
```
the results are stored in ./vis

## Acknowledgements
This code is based on some source code on github:
1. matterport/Mask_RCNN(https://github.com/matterport/Mask_RCNN), an implementation of Mask R-CNN on Python 3, Keras, and TensorFlow. 
2. KingMV/ConvGRU(https://github.com/KingMV/ConvGRU), an implementation of ConvGRU2D on Keras.
