import os
import random
from PIL import Image
import torch
import torch.utils.data
from torchvision import transforms

class YoloFewShotDataset(torch.utils.data.Dataset):
    def __init__(self, yaml_file, transforms=None, num_support_per_class=4):
        """
        YOLO 포맷에 맞춘 few-shot 학습용 데이터셋 클래스.
        :param yaml_file: 데이터셋 구성을 정의한 yaml 파일 경로
        :param transforms: 이미지 변환을 위한 torchvision transforms
        :param num_support_per_class: 각 클래스당 지원 이미지를 몇 개 불러올지 지정
        """
        import yaml

        self.transforms = transforms
        self.num_support_per_class = num_support_per_class

        # YAML 파일에서 데이터셋 정보를 읽어옴
        with open(yaml_file, 'r') as f:
            data_dict = yaml.safe_load(f)

        self.class_names = data_dict['names']  # 클래스 이름 목록을 가져옴
        self.num_classes = len(self.class_names)
        train_dir = data_dict['train']  # 'train' 키를 사용하여 이미지 디렉토리 지정
        # img_dir : train_dir/images, label_dir : train_dir/labels
        self.img_dir = os.path.join(train_dir, 'images')
        self.label_dir = os.path.join(train_dir, 'labels')

        # 각 클래스에 대한 이미지 파일 경로를 저장할 딕셔너리 초기화
        self.class_samples = {i: [] for i in range(self.num_classes)}

        # 이미지와 라벨을 읽어 클래스별로 정리
        self._load_data()

    def _load_data(self):
        """
        이미지와 라벨 파일을 읽어와 각 클래스별로 정리하는 함수.
        """
        for img_file in os.listdir(self.img_dir):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(self.label_dir, label_file)

                # 라벨 파일이 존재하는 경우에만 처리
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            class_id = int(line.split()[0])
                            if class_id < self.num_classes:
                                self.class_samples[class_id].append(img_file)

    def __len__(self):
        # 전체 샘플 수는 쿼리 이미지 수와 동일합니다.
        return sum(len(samples) for samples in self.class_samples.values())

    def _get_support_images(self, class_id):
        """
        특정 클래스에 대한 지원 이미지를 불러오는 함수.
        """
        support_files = random.sample(self.class_samples[class_id], self.num_support_per_class)
        support_images = []
        support_labels = []

        for img_file in support_files:
            img_path = os.path.join(self.img_dir, img_file)
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.label_dir, label_file)

            # 이미지 로드
            image = Image.open(img_path).convert('RGB')
            if self.transforms:
                image = self.transforms(image)
            support_images.append(image)

            # 라벨 로드
            boxes = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.split()
                        class_id = int(parts[0])
                        bbox = list(map(float, parts[1:]))
                        boxes.append([class_id] + bbox)
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 5), dtype=torch.float32)
            support_labels.append(boxes)

        return support_images, support_labels

    def __getitem__(self, idx):
        """
        인덱스에 따라 쿼리 이미지와 지원 세트를 반환하는 메서드.
        """
        # 임의의 쿼리 이미지를 선택
        class_id = idx % self.num_classes
        img_index = idx // self.num_classes
        img_file = self.class_samples[class_id][img_index]

        # 쿼리 이미지 로드
        img_path = os.path.join(self.img_dir, img_file)
        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)

        # 쿼리 이미지의 라벨 로드
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_file)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.split()
                    class_id = int(parts[0])
                    bbox = list(map(float, parts[1:]))
                    boxes.append([class_id] + bbox)
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 5), dtype=torch.float32)

        # 해당 클래스의 지원 이미지와 레이블 가져오기
        support_images, support_labels = self._get_support_images(class_id)

        return image, boxes, support_images, support_labels