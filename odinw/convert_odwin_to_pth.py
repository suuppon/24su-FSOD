import os
import yaml
import json
import torch
import logging

# 로그 설정
logging.basicConfig(filename='log.txt', level=logging.ERROR, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def category_to_phrase(yaml_path: str = 'odinw/detection/odinw_benchmark35_knowledge_and_gpt3.yaml', category_name: str = None):
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        logging.error(f"YAML file not found: {e}")
        return None
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file: {e}")
        return None

    categories = list(data.keys())
    phrases = {}

    for key in categories:
        phrase = None
        try:
            if (data[key]['def_wiki'] != None) and (data[key]['def_wn'] != None):
                if len(data[key]['def_wiki']) > len(data[key]['def_wn']):
                    phrase = data[key]['def_wiki']
                else:
                    phrase = data[key]['def_wn']
            
            if phrase == '':
                phrase = data[key]['gpt3'][0]
        except KeyError as e:
            logging.error(f"Key error in YAML data for category {key}: {e}")
            continue

        phrases[key] = phrase

    if category_name in categories:
        phrase = phrases.get(category_name, category_name)
        
    if phrase is None:
        phrase = category_name

    return phrase

def extract_file_paths(yaml_root: str = 'odinw/detection/odinw_35'):
    yaml_files = []

    try:
        for root, _, files in os.walk(yaml_root):
            for file in files:
                if file.endswith('.yaml'):
                    file_path = os.path.join(root, file)
                    yaml_files.append(file_path)
    except Exception as e:
        logging.error(f"Error walking through directory {yaml_root}: {e}")
        return [], [], []

    train_ann_files, val_ann_files, test_ann_files = [], [], []

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, 'r') as f:
                data = yaml.safe_load(f)
                register = data["DATASETS"]["REGISTER"]
                train = register['train']
                val = register['val']
                test = register['test']

                train_ann_files.append(train['ann_file'])
                val_ann_files.append(val['ann_file'])
                test_ann_files.append(test['ann_file'])
        except FileNotFoundError as e:
            logging.error(f"Annotation YAML file not found: {e}")
        except KeyError as e:
            logging.error(f"Key error in annotation file: {e}")
        except yaml.YAMLError as e:
            logging.error(f"YAML parsing error: {e}")

    return train_ann_files, val_ann_files, test_ann_files

def data_integrate(train_ann_files, val_ann_files, test_ann_files, data_root: str = '/mnt/vc-nfs/jskim'):
    train_integrated_data = integrate_single_data(train_ann_files, data_root, 'train')
    val_integrated_data = integrate_single_data(val_ann_files, data_root, 'val')
    test_integrated_data = integrate_single_data(test_ann_files, data_root, 'test')

    return train_integrated_data, val_integrated_data, test_integrated_data

def integrate_single_data(ann_files, data_root, data_type):
    integrated_data = []
    
    for idx, ann_file in enumerate(ann_files):
        try:
            ann_path = os.path.join(data_root, ann_file)
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)
        except FileNotFoundError as e:
            logging.error(f"Annotation file not found: {e}")
            continue
        except json.JSONDecodeError as e:
            logging.error(f"Error loading JSON file {ann_file}: {e}")
            continue

        keyword = 'odinw/'
        ann_path_prefix = os.path.dirname(ann_path.split(keyword, 1)[1]) if keyword in ann_path else os.path.dirname(ann_path)

        images = {image['id']: image for image in ann_data.get("images", [])}
        annotations = ann_data.get("annotations", [])
        categories = {category['id']: category['name'] for category in ann_data.get("categories", [])}

        for annotation in annotations:
            img_id = annotation['image_id']
            image = images.get(img_id)
            if image is None:
                logging.error(f"Image ID {img_id} not found in annotation {ann_file}")
                continue

            img_file = os.path.join(image['file_name'])
            img_path_w_prefix = os.path.join(ann_path_prefix, img_file)
            category_id = annotation['category_id']
            category_name = categories.get(category_id, "Unknown")
            bbox = annotation['bbox']
            phrase = category_to_phrase(category_name=category_name)
            if phrase != category_name:
                phrase = category_name + '. ' + category_to_phrase(category_name=category_name)
            else:
                phrase = category_name

            integrated_data.append((img_path_w_prefix, '', bbox, phrase, '', category_name))
            print(f"Image file {img_path_w_prefix} is integrated")
        
            if idx % 5 == 0:
                torch.save(integrated_data, f'data/odinw/{data_type}_partial_{idx}.pth')
        print(f"Annotation file {ann_path} is integrated")
        # Save the integrated data every 5 iterations
        
    return integrated_data

def argparser():
    import argparse
    parser = argparse.ArgumentParser(description='Convert odinw dataset to pth file')
    
    parser.add_argument('--yaml_path', type=str, 
                        default='odinw/detection/odinw_benchmark35_knowledge_and_gpt3.yaml', 
                        help='YAML file path')
    parser.add_argument('--data_root', type=str, default='/mnt/vc-nfs/jskim', 
                        help='Root directory of the odinw dataset')
    args = parser.parse_args()
    return args

def main():
    args = argparser()
    train_ann_files, val_ann_files, test_ann_files = extract_file_paths()

    train_data, val_data, test_data = data_integrate(train_ann_files, val_ann_files, test_ann_files, args.data_root)

    torch.save(train_data, 'data/odinw/train.pth')
    torch.save(val_data, 'data/odinw/val.pth')
    torch.save(test_data, 'data/odinw/test.pth')

if __name__ == '__main__':
    main()