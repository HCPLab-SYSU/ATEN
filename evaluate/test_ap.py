import os
import cv2
import numpy as np

PREDICT_DIR = '/your/path/to/results'

GT_DIR = '/your/path/to/VIP/Human_ids'
BBOX_IOU_THRE = [float(x)/100.0 for x in list(range(50, 100, 5))]
MASK_IOU_THRE = [float(x)/100.0 for x in list(range(50, 100, 5))]


############################################################
#  Masks
############################################################

# compute mask overlap
def compute_mask_iou(mask_gt, masks_pre, mask_gt_area, masks_pre_area):
    """Calculates IoU of the given box with the array of the given boxes.
    mask_gt: [H,W] #一个gt的mask
    masks_pre: [num_instances， height, width] predict Instance masks
    mask_gt_area: the gt_mask_area , int
    masks_pre_area: array of length masks_count.包含了所有预测的mask的sum

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    # print('mask_gt',  mask_gt.shape)
    # print('masks_pre', masks_pre.shape)
    # print('masks_pre_area', len(masks_pre_area))
    # mask_gt (500, 375)
    # masks_pre (1, 500, 375)
    # mask_pre_area 1

    intersection = np.logical_and(mask_gt, masks_pre)
    intersection = np.where(intersection == True, 1, 0).astype(np.uint8)
    intersection = NonZero(intersection)

    # print('intersection  ：', intersection)

    mask_gt_areas = np.full(len(masks_pre_area), mask_gt_area)

    union = mask_gt_areas + masks_pre_area[:] - intersection[:]

    iou = intersection / union

    return iou


# 计算mask中所有非零个数
def NonZero(masks):
    """
    :param masks: [N,h,w]一个三维数组，里面放的是二维的mask数组
    :return: (N) 返回一个长度为N的tuple(元素不能修改)，里面是对应的二维mask 中非零位置数目
    """
    area = []  # 先定义成list，返回时修改
    # print('NonZero masks',masks.shape)
    for i in masks:
        _, a = np.nonzero(i)
        area.append(a.shape[0])
    area = tuple(area)
    return area



def compute_mask_overlaps(masks_pre, masks_gt):
    """Computes IoU overlaps between two sets of boxes.
    masks_pre, masks_gt:
    masks_pre 表示待计算的mask [num_instances,height, width] Instance masks
    masks_gt 表示的是ground truth
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of masks_pre and masks_gt 获得所有非零数值个数
    area1 = NonZero(masks_pre)
    area2 = NonZero(masks_gt)

    # print(masks_pre.shape, masks_gt.shape)(1, 375, 1) (500, 375, 1)

    overlaps = np.zeros((masks_pre.shape[0], masks_gt.shape[0]))
    # print('overlaps ： ',overlaps.shape)
    for i in range(overlaps.shape[1]):
        mask_gt = masks_gt[i]
        # print('overlaps：',overlaps)
        overlaps[:, i] = compute_mask_iou(mask_gt, masks_pre, area2[i], area1)

    return overlaps

def voc_ap(rec, prec, use_07_metric=False):
    """
    Compute VOC AP given precision and recall. If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    Args:
        rec: recall
        prec: precision
        use_07_metric:
    Returns:
        ap: average precision
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        # arange([start, ]stop, [step, ]dtype=None)
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap += p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_mask_ap(image_list,iou_threshold):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Input:
    image_list : all picures' id list
    gt_masks：all mask  [N_pictures,num_inst,H,W]
    pre_masks：all predict masks [N_pictures,num_inst,H,W]
    pred_scores：scores for every predicted mask [N_pre_mask]
    pred_class_ids: the indices of all predicted masks

    Returns:
    mAP: Mean Average Precision
    # precisions: List of precisions at different class score thresholds.
    # recalls: List of recall values at different class score thresholds.
    # overlaps: [pred_masks, gt_masks] IoU overlaps.
    """
    iou_thre_num = len(iou_threshold)
    mAp = np.zeros((iou_thre_num,))

    gt_mask_num = 0
    pre_mask_num = 0
    tp = []
    fp = []
    scores = []
    for i in range(iou_thre_num):
        tp.append([])
        fp.append([])
    

    for image_id in image_list:
        gt_mask = cv2.imread(os.path.join(GT_DIR, image_id[0], '%s.png' % image_id[1]), 0)
        pre_mask = cv2.imread(os.path.join(PREDICT_DIR, image_id[0], 'gray', 'inst_%s.png' % image_id[1]), 0)

        gt_mask, n_gt_inst = convert2evalformat(gt_mask)
        pre_mask, n_pre_inst = convert2evalformat(pre_mask)


        gt_mask_num += n_gt_inst
        pre_mask_num += n_pre_inst

        if n_pre_inst == 0:
            continue

        rfp = open(os.path.join(PREDICT_DIR, image_id[0], 'gray', 'scores_%s.txt' % image_id[1]), 'r')
        items = [x.strip().split() for x in rfp.readlines()]
        rfp.close()
        tmp_scores = [x[0] for x in items]
        scores += tmp_scores
        # Compute IoU overlaps [pred_masks, gt_makss]

        if n_gt_inst == 0:
            for i in range(n_pre_inst):
                for k in range(iou_thre_num):
                    fp[k].append(1)
                    tp[k].append(0)
            continue

        gt_mask = np.stack(gt_mask)
        pre_mask = np.stack(pre_mask)
        overlaps = compute_mask_overlaps(pre_mask, gt_mask)

        # print('overlaps.shape',overlaps.shape)

        max_overlap_ind = np.argmax(overlaps, axis=1)

         # l = len(overlaps[:,max_overlap_ind])
        for i in np.arange(len(max_overlap_ind)):
            max_iou = overlaps[i][max_overlap_ind[i]]
            # print('max_iou :', max_iou)
            for k in range(iou_thre_num):
                if max_iou > iou_threshold[k]:
                    tp[k].append(1)
                    fp[k].append(0)
                else:
                    tp[k].append(0)
                    fp[k].append(1)

    ind = np.argsort(scores)[::-1]

    print("----------------seg---------------------")

    for k in range(iou_thre_num):
        m_tp = tp[k]
        m_fp = fp[k]
        m_tp = np.array(m_tp)
        m_fp = np.array(m_fp)

        m_tp = m_tp[ind]
        m_fp = m_fp[ind]

        m_tp = np.cumsum(m_tp)
        m_fp = np.cumsum(m_fp)
        # print('m_tp : ',m_tp)
        # print('m_fp : ', m_fp)
        recall = m_tp / float(gt_mask_num)
        precition = m_tp / np.maximum(m_fp+m_tp, np.finfo(np.float64).eps)

        # Compute mean AP over recall range
        mAp[k] = voc_ap(recall, precition, False)
        print("IOU Threshold:%.2f, mAP:%f"%(iou_threshold[k], mAp[k]))


    print("averge mAP:", np.mean(mAp))
    print("----------------------------------------")


############################################################
#  Bounding Boxes
############################################################

def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [num_instances, height, width]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    boxes = np.zeros([mask.shape[0], 4], dtype=np.int32)
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        # Bounding box.
        # m : [H,W]
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2])

    # 类型为整数
    return boxes.astype(np.int32)


# np基于CPU
def compute_bbox_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'   ground_truth box
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_bbox_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].

    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):#先对gt做处理计算iou
        box2 = boxes2[i]
        overlaps[:, i] = compute_bbox_iou(box2, boxes1, area2[i], area1)#对box_pre的处理在compute_iou函数中进行
    return overlaps


def compute_bbox_ap(image_list, iou_threshold):
    """Compute Average Precision at a set IoU threshold (default 0.5).
    Input:
    image_list : all picures' id list

    Returns:
    mAP: Mean Average Precision
    # precisions: List of precisions at different class score thresholds.
    # recalls: List of recall values at different class score thresholds.
    # overlaps: [pred_masks, gt_masks] IoU overlaps.
    """
    iou_thre_num = len(iou_threshold)
    mAp = np.zeros((iou_thre_num,))
    gt_bbox_num = 0
    pre_bbox_num = 0
    tp = []
    fp = []
    scores = []
    for i in range(iou_thre_num):
        tp.append([])
        fp.append([])


    for image_id in image_list:
        gt_mask = cv2.imread(os.path.join(GT_DIR, image_id[0],'%s.png' % image_id[1]), 0)
        pre_mask = cv2.imread(os.path.join(PREDICT_DIR, image_id[0], 'gray', 'inst_%s.png' % image_id[1]), 0)

        gt_mask, n_gt_inst = convert2evalformat(gt_mask)
        pre_mask, n_pre_inst = convert2evalformat(pre_mask)

        gt_bbox_num += n_gt_inst
        pre_bbox_num += n_pre_inst

        if not pre_mask:
            continue

        rfp = open(os.path.join(PREDICT_DIR, image_id[0], 'gray', 'scores_%s.txt' % image_id[1]), 'r')
        items = [x.strip().split(' ') for x in rfp.readlines()]
        rfp.close()
        tmp_scores = [x[0] for x in items]
        
        scores += tmp_scores

        # Compute IoU overlaps [pred_masks, gt_makss]

        if not gt_mask :
            for i in range(n_pre_inst):
                for k in range(iou_thre_num):
                    fp[k].append(1)
                    tp[k].append(0)
            continue

        gt_mask = np.stack(gt_mask)
        pre_mask = np.stack(pre_mask)

        gt_bbox = extract_bboxes(gt_mask)
        pre_bbox = [x[1:5] for x in items]
        pre_bbox = np.array(pre_bbox, dtype=np.int32)

        # print('pre_bbox .shape', pre_bbox.shape)
        # print('gt_bbox .shape', gt_bbox.shape)

        overlaps = compute_bbox_overlaps(pre_bbox, gt_bbox)
        
        # print('overlaps.shape :',overlaps.shape)

        bbox_overlap_ind = np.argmax(overlaps, axis=1)

         # l = len(overlaps[:,max_overlap_ind])
        for i in np.arange(len(bbox_overlap_ind)):
            max_iou = overlaps[i][bbox_overlap_ind[i]]
            # print('max_iou :',max_iou)
            for k in range(iou_thre_num):
                if max_iou > iou_threshold[k]:
                    tp[k].append(1)
                    fp[k].append(0)
                else:
                    tp[k].append(0)
                    fp[k].append(1)

    ind = np.argsort(scores)[::-1]
    print("----------------detect------------------")
    for k in range(iou_thre_num):
        m_tp = tp[k]
        m_fp = fp[k]
        m_tp = np.array(m_tp)
        m_fp = np.array(m_fp)

        m_tp = m_tp[ind]
        m_fp = m_fp[ind]

        m_tp = np.cumsum(m_tp)
        m_fp = np.cumsum(m_fp)
        # print('m_tp : ',m_tp)
        # print('m_fp : ', m_fp)
        recall = m_tp / float(gt_bbox_num)
        precition = m_tp / np.maximum(m_fp+m_tp, np.finfo(np.float64).eps)

        # Compute mean AP over recall range
        mAp[k] = voc_ap(recall, precition, False)
        print("IOU Threshold:%.2f, mAP:%f"%(iou_threshold[k], mAp[k]))


    print("averge mAP:", np.mean(mAp))
    print("----------------------------------------")





def convert2evalformat(inst_id_map):
    """

    :param inst_id_map:[h, w]
    :return: masks:[instances,h, w]
    """
    masks = []

    inst_ids = np.unique(inst_id_map)
    # print("inst_ids:", inst_ids)
    background_ind = np.where(inst_ids == 0)[0]
    inst_ids = np.delete(inst_ids, background_ind)
    for i in inst_ids:
        im_mask = (inst_id_map == i).astype(np.uint8)

        masks.append(im_mask)

    return masks, len(inst_ids)


if __name__ == '__main__':
    print("result of", PREDICT_DIR)
    

    image_list = []
    for vid in os.listdir(PREDICT_DIR):
        for img in os.listdir(os.path.join(PREDICT_DIR, vid, 'gray')):
            j = img.find('_')
            if img[:j] == 'inst':
                image_list.append([vid, img[j+1:-4]])


#    compute_bbox_ap(image_list, BBOX_IOU_THRE)

    compute_mask_ap(image_list, MASK_IOU_THRE)




