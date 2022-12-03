import numpy as np
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from image_dataset import GameDataset

def update_feature(args,model,device):
    if os.path.exists(args.image_feature) is False:
        os.mkdir(args.image_feature)
    # davice and transform
    data_transform = transforms.Compose([
        transforms.Resize((args.image_width, args.image_height)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset
    image_dataset = GameDataset(IMAGE_DIR=args.image_folder, num_images=-1, transform=data_transform, target_transform=None)

    dataloader = DataLoader(dataset=image_dataset,batch_size=args.batch_size,shuffle=False)

    for i,img in enumerate(dataloader,0):
        feature = model(img).cpu().detach().numpy().flatten()
        np.save(os.path.join(args.image_feature,image_dataset.file_list[i].split('.')[0].split('/')[-1]),feature)

def search_image(args,model,device):
    print('test')
