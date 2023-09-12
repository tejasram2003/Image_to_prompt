import argparse
from utils import *
import pickle
import os
import torch
from get_loader import get_loader

parser = argparse.ArgumentParser()

parser.add_argument("--input_file", type=str, help="Path to the input file")
parser.add_argument("--model", type=str, help="Path to the model", default="model.pth")

args = parser.parse_args()

input_file = args.input_file
model_path = args.model

with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
train_loader, dataset = get_loader(
        root_folder=r"Pytorch\flickr8k\Images",
        annotation_file=r"Pytorch\flickr8k\captions.txt",
        transform=transform,
        num_workers=2,
    )
test_img = transform(Image.open(input_file).convert("RGB")).unsqueeze(0)

print(predict(model, device, dataset,test_img))
