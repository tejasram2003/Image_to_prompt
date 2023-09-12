import os
import sys
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse
import pickle
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
from Pytorch.get_loader import get_loader
from Pytorch.utils import *
import torchvision.transforms as transforms
from Pytorch.utils import save_checkpoint, load_checkpoint, print_examples
from Pytorch.get_loader import get_loader
from Pytorch.model import CNNtoRNN  
from Pytorch import model

sys.modules['model'] = model

# Create your views here.

def home(request, *args, **kwargs ):
    if request.method == "POST":
        file = request.FILES['file']
        result = get_prompt(file)
        print('Prediction successful')
        print(result)
        return JsonResponse({"result": result})
    return render(request, 'home.html')

def get_prompt(file):
    with open('model.pth','rb') as modelfile:
        trained_model = pickle.load(modelfile)
    print("model loaoded")
    
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
    img = transform(Image.open(BytesIO(file.read())).convert("RGB")).unsqueeze(0)
    print("img loaded")
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
    print("Calling predict function")
    return predict(trained_model, device, dataset,img)
