import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# Custom collate function for few-shot object detection
def collate_fn(batch):
    """
    커스텀 collate 함수는 각 배치의 쿼리와 지원 이미지를 동일한 크기로 패딩하고,
    바운딩 박스와 클래스 레이블을 텐서로 변환하여 묶습니다.
    """
    # Query images and labels
    query_images = []
    query_boxes = []
    
    # Support images and labels for each query
    support_images_all = []
    support_boxes_all = []

    for query_image, query_box, support_images, support_boxes in batch:
        # Query images and labels
        query_images.append(query_image)
        query_boxes.append(query_box)
        
        # Support images and labels
        support_images_batch = []
        support_boxes_batch = []
        
        for support_image, support_box in zip(support_images, support_boxes):
            support_images_batch.append(support_image)
            support_boxes_batch.append(support_box)
        
        support_images_all.append(support_images_batch)
        support_boxes_all.append(support_boxes_batch)

    # Convert lists to tensors (use padding where necessary)
    query_images = torch.stack(query_images)  # Shape: (B, C, H, W)
    
    # Padding boxes and converting to tensors
    max_boxes = max([box.shape[0] for box in query_boxes])
    query_boxes_padded = [torch.cat([box, torch.zeros(max_boxes - box.shape[0], 5)], dim=0) for box in query_boxes]
    query_boxes_padded = torch.stack(query_boxes_padded)
    
    # Support set processing
    support_images_tensors = []
    support_boxes_tensors = []

    for support_images, support_boxes in zip(support_images_all, support_boxes_all):
        # Support images tensor
        support_images_tensors.append(torch.stack(support_images))
        
        # Support boxes tensor with padding
        max_support_boxes = max([box.shape[0] for box in support_boxes])
        support_boxes_padded = [torch.cat([box, torch.zeros(max_support_boxes - box.shape[0], 5)], dim=0) for box in support_boxes]
        support_boxes_tensors.append(torch.stack(support_boxes_padded))

    # Stacking all support images and boxes for the batch
    support_images_tensors = torch.stack(support_images_tensors)
    support_boxes_tensors = torch.stack(support_boxes_tensors)

    return query_images, query_boxes_padded, support_images_tensors, support_boxes_tensors
