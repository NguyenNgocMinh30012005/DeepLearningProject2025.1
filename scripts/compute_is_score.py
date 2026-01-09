#!/usr/bin/env python3
"""
Compute Inception Score (IS) for generated images
"""
import argparse
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import inception_v3
from scipy.stats import entropy
import warnings
warnings.filterwarnings('ignore')

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = []
        
        # Collect all images
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def inception_score(imgs, cuda=True, batch_size=32, resize=True, splits=10):
    """
    Compute the inception score of generated images.
    
    Args:
        imgs: Images to compute score for (DataLoader)
        cuda: whether or not to run on GPU
        batch_size: batch size for feeding into Inception v3
        splits: number of splits
    
    Returns:
        mean and std of IS score
    """
    device = torch.device('cuda' if cuda and torch.cuda.is_available() else 'cpu')
    
    # Load inception model
    print("Loading Inception v3 model...")
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model = inception_model.to(device)
    
    # Get predictions
    print("Computing predictions...")
    preds = []
    
    for batch_idx, batch in enumerate(imgs):
        batch = batch.to(device)
        with torch.no_grad():
            pred = inception_model(batch)
        
        # Apply softmax to get probabilities
        pred = torch.nn.functional.softmax(pred, dim=1).cpu().numpy()
        preds.append(pred)
        
        if (batch_idx + 1) % 10 == 0:
            print(f"Processed {(batch_idx + 1) * batch_size} images")
    
    preds = np.concatenate(preds, axis=0)
    
    # Compute the mean kl-div
    print("Computing IS...")
    split_scores = []
    N = preds.shape[0]
    
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)

def main():
    parser = argparse.ArgumentParser(description='Compute Inception Score')
    parser.add_argument('--image_dir', type=str, required=True,
                        help='Path to generated images directory')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for processing')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--splits', type=int, default=10,
                        help='Number of splits for IS computation')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Inception Score (IS) Computation")
    print("=" * 80)
    print(f"Generated images: {args.image_dir}")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Splits: {args.splits}")
    print()
    
    # Check if images exist
    if not os.path.exists(args.image_dir):
        print(f"Error: Directory {args.image_dir} not found!")
        return
    
    # Prepare transforms
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Load dataset
    print("Loading images...")
    dataset = ImageDataset(args.image_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, 
                           shuffle=False, num_workers=4, pin_memory=True)
    
    # Compute IS
    use_cuda = args.device == 'cuda' and torch.cuda.is_available()
    mean, std = inception_score(dataloader, cuda=use_cuda, 
                               batch_size=args.batch_size, splits=args.splits)
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Inception Score: {mean:.4f} ± {std:.4f}")
    print("=" * 80)
    
    # Save results
    result_file = os.path.join(os.path.dirname(args.image_dir), 'inception_score.txt')
    with open(result_file, 'w') as f:
        f.write(f"Inception Score: {mean:.4f} ± {std:.4f}\n")
        f.write(f"Generated images: {args.image_dir}\n")
        f.write(f"Total images: {len(dataset)}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Splits: {args.splits}\n")
    
    print(f"\nResults saved to: {result_file}")

if __name__ == '__main__':
    main()
