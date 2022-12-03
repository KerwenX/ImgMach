import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from image_dataset import GameDataset
import tqdm
import argparse
import sys
from utils import update_feature,search_image

def parse_args(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('--image_folder',required=True,type=str,help='input image folder')
    parser.add_argument('--image_feature',default='feature')
    parser.add_argument('--image_width',required=True,type=int)
    parser.add_argument('--image_height',required=True,type=int)
    parser.add_argument('--batch_size',default=1,required=True,type=int)
    parser.add_argument('--output',default='output')
    parser.add_argument('--eval_only',default=0,type=int)
    parser.add_argument('--target_file',default='ia_300000010.jpg',help='The picture for search')
    parser.add_argument('--model',default='resnet50w5')

    args = parser.parse_args(sys.argv[1:] if args is None else args)
    return args

def main(args):
    print(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Model
    model = torch.hub.load('facebookresearch/swav:main', args.model).to(device)
    model.eval()
    if not args.eval_only:
        # update the images features
        update_feature(args=args,model=model,device=device)
    search_image(args,model=model,device=device)



if __name__ == '__main__':
    main(parse_args())

