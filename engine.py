# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Train and eval functions used in main.py
"""
import math
import os
import sys
import torch
import torch.distributed as dist
from torchvision.ops import box_iou

from tqdm import tqdm
from typing import Iterable

import utils.misc as utils
import utils.loss_utils as loss_utils
import utils.eval_utils as eval_utils
from models.clip import clip
from utils.misc import NestedTensor
import utils.drawing_utils as drawing_utils

from transformers import BertTokenizer


import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import Image, display

def train_one_epoch(args, model: torch.nn.Module, data_loader: Iterable, 
                    optimizer: torch.optim.Optimizer, device: torch.device, 
                    epoch: int, max_norm: float = 0):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    weight_contrast = args.weight_contrast

    iter = epoch * len(data_loader)
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        ( img_data, text_data, target ,tem_imgs, tem_txts, _, category, tem_cat)= batch

        # Copy all tensors to GPU
        img_data = img_data.to(device)
        target = target.to(device)
        if args.model_type == "ResNet":
            text_data = text_data.to(device)
        else:
            text_data = clip.tokenize(text_data).to(device)
            
        # tem_imgs와 tem_txts는 리스트이므로, 각 NestedTensor를 GPU로 이동시킴
        tem_imgs = [tmpl.to(device) for tmpl in tem_imgs]
        tem_txts = [tmpl.to(device) for tmpl in tem_txts]

        # model forward
        output = model(img_data, text_data,tem_imgs, tem_txts, category, tem_cat)
        if output is None:
            print("Error: Model returned None.")

        if args.contrastive_loss == 1:
          loss_dict = loss_utils.trans_vg_contrast(output, target, weight_contrast)
          losses = sum(loss_dict[k] for k in loss_dict.keys())
        else :
          output = output[0]
          loss_dict = loss_utils.trans_vg_loss(output, target)
          losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v
                                      for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        
        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        iter = iter + 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    weight_contrast = args.weight_contrast

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Eval:'

    for batch in metric_logger.log_every(data_loader, 10, header):
        ( img_data, text_data, target ,tem_imgs, tem_txts, tem_bboxes, category,tem_cat ) = batch
        if args.model_type == "ResNet":
            batch_size = img_data.tensors.size(0)
        else:
            batch_size = img_data.size(0)
        # copy to GPU
        img_data = img_data.to(device)
        # tem_imgs와 tem_txts는 리스트이므로, 각 NestedTensor를 GPU로 이동시킴
        tem_imgs = [tmpl.to(device) for tmpl in tem_imgs]
        tem_txts = [tmpl.to(device) for tmpl in tem_txts]
        tem_bboxes = [tmpl.to(device) for tmpl in tem_bboxes]

        if args.model_type == "ResNet":
            text_data = text_data.to(device)
        else:
            text_data = clip.tokenize(text_data).to(device)
        target = target.to(device)

        pred_boxes = model(img_data, text_data, tem_imgs, tem_txts, category,tem_cat)

        if args.contrastive_loss == 1:
          loss_dict = loss_utils.trans_vg_contrast(pred_boxes, target,weight_contrast)
          losses = sum(loss_dict[k] for k in loss_dict.keys())
          pred_boxes = pred_boxes[0]
        else :
          pred_boxes = pred_boxes[0]
          loss_dict = loss_utils.trans_vg_loss(pred_boxes, target)
          losses = sum(loss_dict[k] for k in loss_dict.keys())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {k: v
                                      for k, v in loss_dict_reduced.items()}
        losses_reduced_unscaled = sum(loss_dict_reduced_unscaled.values())
        loss_value = losses_reduced_unscaled.item()

        miou, accu = eval_utils.trans_vg_eval_val(pred_boxes, target)

        metric_logger.update(loss=loss_value, **loss_dict_reduced_unscaled)
        metric_logger.update_v2('miou', torch.mean(miou), batch_size)
        metric_logger.update_v2('accu', accu, batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    print("Averaged stats:", metric_logger)
    return stats

# BERT tokenizer load
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

@torch.no_grad()
def evaluate(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()
    weight_contrast = args.weight_contrast
    pred_box_list = []
    gt_box_list = []

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        (img_data, text_data, target, tem_imgs, tem_txts, tem_bboxes, category, tem_cat) = batch
        batch_size = img_data.tensors.size(0)

        
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        target = target.to(device)
        tem_imgs = [tmpl.to(device) for tmpl in tem_imgs]
        tem_txts = [tmpl.to(device) for tmpl in tem_txts]
        tem_bboxes = [tmpl.to(device) for tmpl in tem_bboxes]

        # Model prediction
        output = model(img_data, text_data, tem_imgs, tem_txts, category, tem_cat)
        
        if args.contrastive_loss == 1:
          loss_dict = loss_utils.trans_vg_contrast(output, target,weight_contrast)
          losses = sum(loss_dict[k] for k in loss_dict.keys())
          output = output[0]
        else :
          output = output[0]
          loss_dict = loss_utils.trans_vg_loss(output, target)
          losses = sum(loss_dict[k] for k in loss_dict.keys())

        # Save predictions and ground truth
        pred_box_list.append(output.cpu())
        gt_box_list.append(target.cpu())

        # 이미지 시각화 
        if isinstance(text_data, NestedTensor):
            text_data = text_data.tensors

        # token 화 된 text decoding 수행 
        current_text_tokens = text_data[0].cpu().numpy()
        current_text = tokenizer.decode(current_text_tokens, skip_special_tokens=True) 

        if batch_idx % 1 == 0:
            output_dir_path = args.output_dir
            save_path = f"batch_{batch_idx}_eval_image.png"
            drawing_utils.draw_bounding_boxes(img_data.tensors[0], tem_imgs, output[0], current_text, target[0], save_path=save_path)

    pred_boxes = torch.cat(pred_box_list, dim=0)
    gt_boxes = torch.cat(gt_box_list, dim=0)

    total_num = gt_boxes.shape[0]
    accu_num, AP = eval_utils.trans_vg_eval_test(pred_boxes, gt_boxes)

    result_tensor = torch.tensor([accu_num, total_num]).to(device)
    
    torch.cuda.synchronize()
    dist.all_reduce(result_tensor)

    accuracy = float(result_tensor[0]) / float(result_tensor[1])
    
    return accuracy, AP


def draw_bounding_boxes_inference(image, pred_boxes, text, gt_boxes=None, figsize=(10, 10), save_path="output_image.png"):
    fig, ax = plt.subplots(1, figsize=figsize)

    # default로 쓰이는 image 평균과 표준편차
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # image 역정규화
    img = drawing_utils.denormalize(image, mean, std)
    
    img = torch.clamp(img, 0, 1)
    
    height, width = image.shape[1], image.shape[2]

    # 시각화
    ax.imshow(img.permute(1, 2, 0).cpu().numpy())

    # 예측 바운딩 박스 그리기 (Center-Width-Height 형식으로 출력되므로 Xmin-Xmax-Ymin-Ymax로 변환)
    if pred_boxes.numel() == 4:
        center_x, center_y, box_width, box_height = pred_boxes
        xmin = (center_x - (box_width / 2)) * width
        xmax = (center_x + (box_width / 2)) * width
        ymin = (center_y - (box_height / 2)) * height
        ymax = (center_y + (box_height / 2)) * height

        # 좌표로부터 너비와 높이 계산
        width_rect = xmax - xmin
        height_rect = ymax - ymin

        # 예측된 바운딩 박스 (빨간색)
        rect = patches.Rectangle((xmin.cpu(), ymin.cpu()), width_rect.cpu(), height_rect.cpu(),
                                 linewidth=3, edgecolor='r', facecolor='none', label="Prediction")
        ax.add_patch(rect)

    # 텍스트 크기와 위치 조정
    ax.text(20, 20, f"Text: {text}", color='white', fontsize=16, bbox=dict(facecolor='black', alpha=0.7))

    # 범례 추가
    ax.legend(loc="upper right")

    # 이미지 저장
    plt.savefig(save_path)
    plt.close(fig)

# inference 코드 
@torch.no_grad()
def inference(args, model: torch.nn.Module, data_loader: Iterable, device: torch.device):
    model.eval()

    for batch_idx, batch in enumerate(tqdm(data_loader)):
        (img_data, text_data, target, tem_imgs, tem_txts, tem_bboxes, category, tem_cat) = batch
        batch_size = img_data.tensors.size(0)

        
        img_data = img_data.to(device)
        text_data = text_data.to(device)
        tem_imgs = [tmpl.to(device) for tmpl in tem_imgs]
        tem_txts = [tmpl.to(device) for tmpl in tem_txts]
        tem_bboxes = [tmpl.to(device) for tmpl in tem_bboxes]

        # Model prediction
        output = model(img_data, text_data, tem_imgs, tem_txts, category, tem_cat)

      
        if isinstance(text_data, NestedTensor):
            text_data = text_data.tensors

        
        current_text_tokens = text_data[0].cpu().numpy()
        current_text = tokenizer.decode(current_text_tokens, skip_special_tokens=True)

        if batch_idx % 20 == 0:
            save_path = f"batch_{batch_idx}_inference_image.png"
            draw_bounding_boxes_inference(img_data.tensors[0], output[0], current_text, save_path=save_path)