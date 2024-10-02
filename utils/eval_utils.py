import torch
import numpy as np

from utils.box_utils import bbox_iou, xywh2xyxy


def trans_vg_eval_val(pred_boxes, gt_boxes):
    batch_size = pred_boxes.shape[0]
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu = torch.sum(iou >= 0.5) / float(batch_size)

    return iou, accu

def trans_vg_eval_test(pred_boxes, gt_boxes):
    pred_boxes = xywh2xyxy(pred_boxes)
    pred_boxes = torch.clamp(pred_boxes, 0, 1)
    gt_boxes = xywh2xyxy(gt_boxes)
    iou = bbox_iou(pred_boxes, gt_boxes)
    accu_num = torch.sum(iou >= 0.5)
    AP = calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5)

    return accu_num, AP

def calculate_ap(pred_boxes, gt_boxes, iou_threshold=0.5):
    """
    Calculate the Average Precision (AP) for a single class without confidence scores.

    Args:
        pred_boxes (Tensor): Predicted boxes, shape [num_predictions, 4] (x1, y1, x2, y2).
        gt_boxes (Tensor): Ground truth boxes, shape [num_ground_truths, 4] (x1, y1, x2, y2).
        iou_threshold (float): IoU threshold to consider a detection as True Positive.

    Returns:
        float: Average Precision (AP).
    """
    num_pred = pred_boxes.size(0)
    num_gt = gt_boxes.size(0)

    # Initialize True Positives (TP) and False Positives (FP)
    tp = torch.zeros(num_pred)
    fp = torch.zeros(num_pred)
    gt_detected = torch.zeros(num_gt)  # Track which GT boxes are detected

    # Loop through each prediction and match with ground truth based on IoU
    for pred_idx in range(num_pred):
        pred_box = pred_boxes[pred_idx, :4]
        max_iou = 0
        match_gt_idx = -1

        # Compare with each ground truth box
        for gt_idx in range(num_gt):
            if gt_detected[gt_idx] == 1:  # Skip already detected GT boxes
                continue

            iou = bbox_iou(pred_box.unsqueeze(0), gt_boxes[gt_idx].unsqueeze(0)).item()
            if iou > max_iou:
                max_iou = iou
                match_gt_idx = gt_idx

        # Determine if it is a True Positive or False Positive
        if max_iou >= iou_threshold:
            tp[pred_idx] = 1
            gt_detected[match_gt_idx] = 1  # Mark this GT box as detected
        else:
            fp[pred_idx] = 1

    # Calculate cumulative True Positives and False Positives
    cumulative_tp = torch.cumsum(tp, dim=0)
    cumulative_fp = torch.cumsum(fp, dim=0)

    # Calculate Precision and Recall
    precision = cumulative_tp / (cumulative_tp + cumulative_fp + 1e-8)
    recall = cumulative_tp / num_gt

    # Ensure that precision is monotonically decreasing
    for i in range(num_pred - 1, 0, -1):
        precision[i - 1] = torch.max(precision[i - 1], precision[i])

    # Insert 0 at the start of recall and precision for proper interpolation
    precision = torch.cat((torch.tensor([1.0]), precision))
    recall = torch.cat((torch.tensor([0.0]), recall))

    # Calculate Average Precision (AP) using trapezoidal rule
    ap = torch.trapz(precision, recall).item()

    return ap