import numpy as np
import os

import torchvision
from torch.utils.data import DataLoader
from image_dataset import GameDataset
from PIL import Image
from tqdm import tqdm
import operator

def update_feature(args,model,device,data_transform):
    if os.path.exists(args.image_feature) is False:
        os.mkdir(args.image_feature)
    # Dataset
    image_dataset = GameDataset(IMAGE_DIR=args.image_folder, num_images=-1, transform=data_transform, target_transform=None)

    dataloader = DataLoader(dataset=image_dataset,batch_size=args.batch_size,shuffle=False)

    for i,img in enumerate(dataloader,0):
        feature = model(img).cpu().detach().numpy().flatten()
        np.save(os.path.join(args.image_feature,image_dataset.file_list[i].split('.')[0].split('/')[-1]),feature)

def search_image(args,model,device,data_transform):
    print("target file name :",{args.target_file})
    source_image = Image.open(args.target_file)
    assert isinstance(data_transform,torchvision.transforms.Compose)
    source_image = data_transform(source_image).unsqueeze(0)
    feature = model(source_image).cpu().detach().numpy().flatten()
    features_folder = args.image_feature
    feature_list = os.listdir(features_folder)
    similarity = {}
    for i,filename in enumerate(tqdm(feature_list)):
        if not filename.endswith('npy'):
            continue
        fts = np.load(os.path.join(args.image_feature,filename)).flatten()
        cos_sim = np.dot(feature,fts.T)/ (np.linalg.norm(feature)*np.linalg.norm(fts))
        similarity[filename] = cos_sim
    sort_similarity = sorted(similarity.items(),key=operator.itemgetter(1),reverse=True)
    print("The most similarity file is ",sort_similarity[0][0])
    return sort_similarity[0][0]


