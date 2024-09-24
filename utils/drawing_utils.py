import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# image tensor 정규화 된 상태에서 다시역정규화 시킴 
def denormalize(image_tensor, mean, std):
    
    img = image_tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)  
    return img

# bounding box + query text 출력 
def draw_bounding_boxes(image, pred_boxes, text, gt_boxes=None, figsize=(10, 10), save_path="output_image.png"):
    fig, ax = plt.subplots(1, figsize=figsize)

    # default로 쓰이는 image 평균과 표준편차 
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # image 역정규화 
    img = denormalize(image, mean, std)
    
    img = torch.clamp(img, 0, 1)
    
    height, width = image.shape[1], image.shape[2]

    # 시각화 
    ax.imshow(img.permute(1, 2, 0).cpu().numpy())

    # bounding box 그리기  (Center-Width-Height 형식으로 출력되서 refcocog 에 맞게  Xmin-Xmax-Ymin-Ymax로 변환)
    if pred_boxes.numel() == 4:
        center_x, center_y, box_width, box_height = pred_boxes
        xmin = (center_x - (box_width / 2)) * width
        xmax = (center_x + (box_width / 2)) * width
        ymin = (center_y - (box_height / 2)) * height
        ymax = (center_y + (box_height / 2)) * height

        # 좌표로부터 너비와 높이 계산
        width_rect = xmax - xmin
        height_rect = ymax - ymin

        
        rect = patches.Rectangle((xmin.cpu(), ymin.cpu()), width_rect.cpu(), height_rect.cpu(),
                                 linewidth=3, edgecolor='r', facecolor='none', label="Prediction")
        ax.add_patch(rect)

  
    if gt_boxes is not None and gt_boxes.numel() == 4:
        center_x, center_y, box_width, box_height = gt_boxes
        xmin = (center_x - (box_width / 2)) * width
        xmax = (center_x + (box_width / 2)) * width
        ymin = (center_y - (box_height / 2)) * height
        ymax = (center_y + (box_height / 2)) * height

        width_rect = xmax - xmin
        height_rect = ymax - ymin

        rect = patches.Rectangle((xmin.cpu(), ymin.cpu()), width_rect.cpu(), height_rect.cpu(),
                                 linewidth=2, edgecolor='g', facecolor='none', linestyle='--', label="Ground Truth")
        ax.add_patch(rect)

    # 텍스트 크기와 위치 조정
    ax.text(20, 20, f"Text: {text}", color='white', fontsize=16, bbox=dict(facecolor='black', alpha=0.7))

    
    ax.legend(loc="upper right")

   
    plt.savefig(save_path)
    plt.close(fig)

    #display(Image(filename=save_path))
