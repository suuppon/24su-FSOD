import os
import yaml
import json
import torch

def category_to_phrase(yaml_path: str = 'odinw/detection/odinw_benchmark35_knowledge_and_gpt3.yaml', category_name: str = None):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    categories = list(data.keys())
    phrases = {}
    
    # 카테고리와 매핑된 phrase를 생성
    for key in categories:
        phrase = None
        if (data[key]['def_wiki'] != None) and (data[key]['def_wn'] != None):
            if len(data[key]['def_wiki']) > len(data[key]['def_wn']):
                phrase = data[key]['def_wiki']
            else:
                phrase = data[key]['def_wn']
        
        if phrase == '':
            phrase = data[key]['gpt3'][0]
        
        # 카테고리와 phrase를 딕셔너리로 매핑
        phrases[key] = phrase

    # 입력받은 카테고리 이름이 있으면 매핑된 phrase를 출력
    if category_name in categories:
        phrase = phrases[category_name]
    
    if phrase != None:
        return phrase
    else:
        return category_name
    
    
def extract_file_paths(yaml_root: str = 'odinw/detection/odinw_35'):
    # data_root 내의 yaml 파일 순회
    yaml_files = []
    
    for root, _, files in os.walk(yaml_root):
        for file in files:
            if file.endswith('.yaml'):
                file_path = os.path.join(root, file)
                yaml_files.append(file_path)
    
    train_ann_files = []
    val_ann_files = []
    test_ann_files = []
    
    for yaml_file in yaml_files:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
            register = data["DATASETS"]["REGISTER"]
            train = register['train']
            val = register['val']
            test = register['test']
            
            train_ann_file = train['ann_file']
            val_ann_file = val['ann_file']
            test_ann_file = test['ann_file']
            
            train_ann_files.append(train_ann_file)
            val_ann_files.append(val_ann_file)
            test_ann_files.append(test_ann_file)
        
        
    return train_ann_files, val_ann_files, test_ann_files


def data_integrate(train_ann_files, 
                   val_ann_files, 
                   test_ann_files, 
                   data_root: str = '/mnt/vc-nfs/jskim'):
    
    train_integrated_data = integrate_single_data(train_ann_files, data_root)
    val_integrated_data = integrate_single_data(val_ann_files, data_root)
    test_integrated_data = integrate_single_data(test_ann_files, data_root)
    
    return train_integrated_data, val_integrated_data, test_integrated_data

def integrate_single_data(ann_files, data_root):
    
    integrated_data = []
    
    # 모든 train annotation 파일에 대해 처리
    for ann_file in ann_files:
        ann_path = os.path.join(data_root, ann_file)
        
        
        with open(ann_path, 'r') as f:
            ann_data = json.load(f)
        
        keyword = 'odinw'
        if keyword in ann_path:
            ann_path_prefix = os.path.dirname(ann_path.split(keyword, 1)[1])  # 'odinw' 이후의 경로에서 JSON 파일명 제거
        else:
            ann_path_prefix = os.path.dirname(ann_path) 
        
        images = {image['id']: image for image in ann_data["images"]}  # 이미지 ID를 키로 하는 딕셔너리 생성
        annotations = ann_data["annotations"]
        categories = {category['id']: category['name'] for category in ann_data["categories"]}
        
        # 각 annotation에 대한 정보 처리
        for annotation in annotations:
            img_id = annotation['image_id']
            image = images[img_id]
            img_file = os.path.join(image['file_name'])  # 이미지 경로에 data_root 추가
            img_path_w_prefix = os.path.join(ann_path_prefix, img_file)
            category_id = annotation['category_id']
            category_name = categories[category_id]
            bbox = annotation['bbox']
            phrase = category_to_phrase(category_name=category_name)
            if phrase != category_name:

                phrase = category_name + '. ' + category_to_phrase(category_name=category_name)
            else:
                phrase = category_name
    
            integrated_data.append((img_path_w_prefix, '', bbox, phrase, '', category_name))
        
        
        print("Annotation file {} is integrated".format(ann_path))
    
    return integrated_data
    

def argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Convert odinw dataset to pth file')
    
    parser.add_argument('--yaml_path', type=str, 
                        default='odinw/detection/odinw_benchmark35_knowledge_and_gpt3.yaml', 
                        help='이거 경로 --> https://github.com/Computer-Vision-in-the-Wild/DataDownload/blob/master/detection/odinw_benchmark35_knowledge_and_gpt3.yaml')
    parser.add_argument('--data_root', type=str, default='/mnt/vc-nfs/jskim', 
                        help='odinw 데이터셋이 있는 상위 디렉토리')
    args = parser.parse_args()
    return args

def main():

    train_ann_files, val_ann_files, test_ann_files = extract_file_paths()
    
    train_data, val_data, test_data = data_integrate(train_ann_files, val_ann_files, test_ann_files)
    
    torch.save(train_data, 'data/odinw/train.pth')
    torch.save(val_data, 'data/odinw/val.pth')
    torch.save(test_data, 'data/odinw/test.pth')
    
if __name__ == '__main__':
    main()