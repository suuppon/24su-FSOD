import argparse
import os

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataset_names", default="all", type=str)  # "all" or names joined by comma
argparser.add_argument("--dataset_path", default="odinw", type=str)
args = argparser.parse_args()

root = "https://huggingface.co/GLIPModel/GLIP/resolve/main/odinw_35"

all_datasets = ["AerialMaritimeDrone", "AmericanSignLanguageLetters", "Aquarium", "BCCD", "ChessPieces", "CottontailRabbits", "DroneControl", "EgoHands", "HardHatWorkers", "MaskWearing", "MountainDewCommercial", "NorthAmericaMushrooms", "OxfordPets", "PKLot", "Packages", "PascalVOC", "Raccoon", "ShellfishOpenImages", "ThermalCheetah", "UnoCards", "VehiclesOpenImages", "WildfireSmoke", "boggleBoards", "brackishUnderwater", "dice", "openPoetryVision", "pistols", "plantdoc", "pothole", "selfdrivingCar", "thermalDogsAndPeople", "vector", "websiteScreenshots"]

datasets_to_download = []
if args.dataset_names == "all":
    datasets_to_download = all_datasets
else:
    datasets_to_download = args.dataset_names.split(",")

# 디렉토리가 없을 경우 생성
if not os.path.exists(args.dataset_path):
    os.makedirs(args.dataset_path)

for dataset in datasets_to_download:
    if dataset in all_datasets:
        print("Downloading dataset: ", dataset)
        # wget으로 파일을 지정된 디렉토리에 저장
        os.system("wget " + root + "/" + dataset + ".zip" + " -O " + os.path.join(args.dataset_path, dataset + ".zip"))
        os.system("unzip " + os.path.join(args.dataset_path, dataset + ".zip") + " -d " + args.dataset_path)
        os.system("rm " + os.path.join(args.dataset_path, dataset + ".zip"))
    else:
        print("Dataset not found: ", dataset)