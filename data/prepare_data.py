import os
import argparse
from PIL import Image
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

def prepare_gtsrb(data_dir, output_dir, image_size=64):
    """Preprocess GTSRB dataset: resize images to 64x64 and organize splits."""
    os.makedirs(output_dir, exist_ok=True)
    splits = ['Final_Training', 'Final_Test']
    
    for split in splits:
        split_dir = os.path.join(data_dir, split, 'Images')
        output_split_dir = os.path.join(output_dir, split.lower())
        os.makedirs(output_split_dir, exist_ok=True)
        
        # Read annotations
        csv_file = os.path.join(data_dir, split, f'GT-final_{split.lower()}.csv')
        annotations = pd.read_csv(csv_file, sep=';')
        
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])
        
        for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc=f'Processing {split}'):
            img_path = os.path.join(split_dir, row['Filename'])
            class_id = row['ClassId']
            class_dir = os.path.join(output_split_dir, f'{class_id:05d}')
            os.makedirs(class_dir, exist_ok=True)
            
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            output_path = os.path.join(class_dir, row['Filename'])
            torchvision.utils.save_image(img, output_path)

def prepare_cifar10(output_dir, image_size=64):
    """Download and preprocess CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    
    trainset = torchvision.datasets.CIFAR10(root=output_dir, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=output_dir, train=False, download=True, transform=transform)
    
    os.makedirs(os.path.join(output_dir, 'cifar10', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'cifar10', 'test'), exist_ok=True)
    
    for i, (img, label) in tqdm(enumerate(trainset), total=len(trainset), desc='Processing CIFAR-10 Train'):
        torchvision.utils.save_image(img, os.path.join(output_dir, 'cifar10', 'train', f'{label}_{i}.png'))
    
    for i, (img, label) in tqdm(enumerate(testset), total=len(testset), desc='Processing CIFAR-10 Test'):
        torchvision.utils.save_image(img, os.path.join(output_dir, 'cifar10', 'test', f'{label}_{i}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare GTSRB or CIFAR-10 datasets.')
    parser.add_argument('--dataset', type=str, choices=['gtsrb', 'cifar10'], required=True)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--output_dir', type=str, default='./data/processed')
    parser.add_argument('--image_size', type=int, default=64)
    args = parser.parse_args()
    
    if args.dataset == 'gtsrb':
        prepare_gtsrb(args.data_dir, args.output_dir, args.image_size)
    else:
        prepare_cifar10(args.output_dir, args.image_size)
