import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# image tensor 정규화 된 상태에서 다시역정규화 시킴 
def denormalize(image_tensor, mean, std):
    
    img = image_tensor.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)  
    return img

def draw_bounding_boxes(image, tem_imgs, pred_boxes, text, gt_boxes=None, figsize=(10, 10), save_path="output_image.png"):
    fig, ax = plt.subplots(1, figsize=figsize)

    # default로 쓰이는 image 평균과 표준편차
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])

    # image 역정규화
    img = denormalize(image, mean, std)
    
    img = torch.clamp(img, 0, 1)
    
    height, width = image.shape[1], image.shape[2]
    
    tem_images = []
    
    for tem in tem_imgs:
        tem = tem.tensors[0]
        
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        
        tem = denormalize(tem, mean, std)
        
        tem = torch.clamp(tem, 0, 1).permute(1, 2, 0).cpu().numpy()
        
        tem_images.append(tem)
    
    # 시각화
    ax.imshow(img.permute(1, 2, 0).cpu().numpy())

    # 예측 바운딩 박스 그리기
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

    # 작은 템플릿 이미지들을 하단에 그리기
    spacing = 5  # 이미지 간 간격 설정
    for i, template_img in enumerate(tem_images):
        # 템플릿 이미지를 그릴 위치 계산
        x_offset = 0.1 + i * (0.15 + spacing/100)  # 시작 위치 + 이미지 간격
        y_offset = -0.15  # 메인 이미지 아래쪽에 배치
        
        # 템플릿 이미지를 작은 서브플롯으로 추가
        inset_ax = fig.add_axes([x_offset, y_offset, 0.1, 0.1])  # [left, bottom, width, height]
        inset_ax.imshow(template_img)
        inset_ax.axis('off')  # 축 숨기기

    # 이미지 저장 및 닫기
    plt.savefig(save_path)
    plt.close(fig)
