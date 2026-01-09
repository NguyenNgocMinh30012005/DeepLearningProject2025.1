#!/usr/bin/env python3
"""
Chuẩn bị datasets cho experiments:
1. Split real data -> train/val/test
2. Tạo balanced dataset (real + synthetic)
"""
import os
import shutil
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def count_images_per_class(data_dir):
    """Đếm số ảnh mỗi class"""
    counts = {}
    for class_name in sorted(os.listdir(data_dir)):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            counts[class_name] = len(images)
    return counts

def split_real_dataset(real_dir, output_dir, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Split real dataset thành train/val/test
    """
    print(f"\n{'='*80}")
    print("SPLITTING REAL DATASET")
    print(f"{'='*80}")
    print(f"Source: {real_dir}")
    print(f"Output: {output_dir}")
    print(f"Split: Train={train_ratio}, Val={val_ratio}, Test={1-train_ratio-val_ratio}")
    print()
    
    np.random.seed(seed)
    
    stats = {
        'classes': [],
        'total_train': 0,
        'total_val': 0,
        'total_test': 0
    }
    
    # Tạo output dirs
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(output_dir, split), exist_ok=True)
    
    # Process mỗi class
    class_names = sorted([d for d in os.listdir(real_dir) 
                         if os.path.isdir(os.path.join(real_dir, d))])
    
    for class_name in tqdm(class_names, desc="Splitting classes"):
        class_path = os.path.join(real_dir, class_name)
        
        # Lấy danh sách ảnh
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) == 0:
            print(f"Warning: No images in {class_name}")
            continue
        
        # Split
        train_val, test = train_test_split(
            images, test_size=1-train_ratio-val_ratio, random_state=seed
        )
        train, val = train_test_split(
            train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=seed
        )
        
        # Copy files
        for split, split_images in [('train', train), ('val', val), ('test', test)]:
            split_class_dir = os.path.join(output_dir, split, class_name)
            os.makedirs(split_class_dir, exist_ok=True)
            
            for img in split_images:
                src = os.path.join(class_path, img)
                dst = os.path.join(split_class_dir, img)
                shutil.copy2(src, dst)
        
        stats['classes'].append({
            'name': class_name,
            'total': len(images),
            'train': len(train),
            'val': len(val),
            'test': len(test)
        })
        stats['total_train'] += len(train)
        stats['total_val'] += len(val)
        stats['total_test'] += len(test)
    
    # Save stats
    with open(os.path.join(output_dir, 'split_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Split complete:")
    print(f"  Train: {stats['total_train']} images")
    print(f"  Val: {stats['total_val']} images")
    print(f"  Test: {stats['total_test']} images")
    
    return stats

def create_balanced_dataset(real_dir, synth_dir, output_dir, target_per_class=1024):
    """
    Tạo balanced dataset bằng cách combine real + synthetic
    Mục tiêu: Mỗi class có target_per_class ảnh
    """
    print(f"\n{'='*80}")
    print("CREATING BALANCED DATASET (Real + Synthetic)")
    print(f"{'='*80}")
    print(f"Real dir: {real_dir}")
    print(f"Synth dir: {synth_dir}")
    print(f"Output: {output_dir}")
    print(f"Target per class: {target_per_class}")
    print()
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        'classes': [],
        'total_images': 0,
        'total_real': 0,
        'total_synth': 0,
        'target_per_class': target_per_class
    }
    
    # Lấy danh sách classes từ real data
    class_names = sorted([d for d in os.listdir(real_dir) 
                         if os.path.isdir(os.path.join(real_dir, d))])
    
    for class_name in tqdm(class_names, desc="Creating balanced dataset"):
        real_class_dir = os.path.join(real_dir, class_name)
        synth_class_dir = os.path.join(synth_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        # Lấy real images
        real_images = [f for f in os.listdir(real_class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        n_real = len(real_images)
        
        # Copy all real images
        for img in real_images:
            src = os.path.join(real_class_dir, img)
            dst = os.path.join(output_class_dir, f"real_{img}")
            shutil.copy2(src, dst)
        
        # Tính số synthetic images cần thêm
        n_synth_needed = max(0, target_per_class - n_real)
        
        # Lấy synthetic images
        if os.path.exists(synth_class_dir) and n_synth_needed > 0:
            synth_images = [f for f in os.listdir(synth_class_dir) 
                           if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # Sample synthetic images
            if len(synth_images) >= n_synth_needed:
                selected_synth = np.random.choice(synth_images, n_synth_needed, replace=False)
            else:
                # Nếu không đủ synth, lấy hết và repeat
                selected_synth = synth_images
                if len(synth_images) > 0:
                    n_repeat = n_synth_needed - len(synth_images)
                    repeated = np.random.choice(synth_images, n_repeat, replace=True)
                    selected_synth = list(selected_synth) + list(repeated)
            
            # Copy synthetic images
            for idx, img in enumerate(selected_synth):
                src = os.path.join(synth_class_dir, img)
                dst = os.path.join(output_class_dir, f"synth_{idx:05d}_{img}")
                shutil.copy2(src, dst)
            
            n_synth_used = len(selected_synth)
        else:
            n_synth_used = 0
        
        total_class = n_real + n_synth_used
        
        stats['classes'].append({
            'name': class_name,
            'real': n_real,
            'synth': n_synth_used,
            'total': total_class
        })
        stats['total_images'] += total_class
        stats['total_real'] += n_real
        stats['total_synth'] += n_synth_used
        
        print(f"{class_name}: Real={n_real}, Synth={n_synth_used}, Total={total_class}")
    
    # Save stats
    with open(os.path.join(output_dir, 'balanced_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✓ Balanced dataset created:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Real: {stats['total_real']}")
    print(f"  Synthetic: {stats['total_synth']}")
    
    return stats

def main():
    parser = argparse.ArgumentParser(description='Prepare datasets for experiments')
    parser.add_argument('--real_dir', type=str, required=True,
                       help='Real dataset directory')
    parser.add_argument('--synth_dir', type=str, required=True,
                       help='Synthetic dataset directory')
    parser.add_argument('--output_base', type=str, default='/workspace/experiments_data',
                       help='Base output directory')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                       help='Train split ratio')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                       help='Val split ratio')
    parser.add_argument('--target_per_class', type=int, default=1024,
                       help='Target images per class for balanced dataset')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print(f"{'='*80}")
    print("DATASET PREPARATION")
    print(f"{'='*80}")
    
    # 1. Phân tích dataset gốc
    print("\nAnalyzing original dataset...")
    real_counts = count_images_per_class(args.real_dir)
    print(f"\nReal dataset class distribution:")
    for class_name, count in sorted(real_counts.items(), key=lambda x: x[1]):
        print(f"  {class_name}: {count} images")
    print(f"\nMin: {min(real_counts.values())}, Max: {max(real_counts.values())}")
    print(f"Total: {sum(real_counts.values())} images")
    
    # 2. Split real dataset
    real_split_dir = os.path.join(args.output_base, 'real_split')
    split_stats = split_real_dataset(
        args.real_dir, 
        real_split_dir, 
        args.train_ratio, 
        args.val_ratio, 
        args.seed
    )
    
    # 3. Create balanced dataset cho train
    balanced_train_dir = os.path.join(args.output_base, 'balanced_train')
    balanced_stats = create_balanced_dataset(
        os.path.join(real_split_dir, 'train'),
        args.synth_dir,
        balanced_train_dir,
        args.target_per_class
    )
    
    # 4. Create balanced dataset cho val
    balanced_val_dir = os.path.join(args.output_base, 'balanced_val')
    balanced_val_stats = create_balanced_dataset(
        os.path.join(real_split_dir, 'val'),
        args.synth_dir,
        balanced_val_dir,
        args.target_per_class // 4  # Val dùng ít hơn
    )
    
    print(f"\n{'='*80}")
    print("✓ ALL DATASETS PREPARED")
    print(f"{'='*80}")
    print(f"\nOutput structure:")
    print(f"  {args.output_base}/")
    print(f"    real_split/")
    print(f"      train/  ({split_stats['total_train']} images)")
    print(f"      val/    ({split_stats['total_val']} images)")
    print(f"      test/   ({split_stats['total_test']} images)")
    print(f"    balanced_train/  ({balanced_stats['total_images']} images)")
    print(f"    balanced_val/    ({balanced_val_stats['total_images']} images)")

if __name__ == '__main__':
    main()
