#!/usr/bin/env python3
"""
Train ViT_Tiny với:
- Train/Val trên generated_images
- Test trên dataset_original/PlantVillage
"""
import os
import shutil
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

def prepare_dataset_splits(generated_dir, output_base, val_split=0.2, random_state=42):
    """
    Chia generated_images thành train/val
    
    Args:
        generated_dir: Thư mục chứa generated images (có 15 classes)
        output_base: Thư mục output để lưu train/val
        val_split: Tỷ lệ validation (default: 0.2 = 20%)
        random_state: Random seed
    """
    print("="*80)
    print("PREPARING DATASET SPLITS")
    print("="*80)
    print(f"Source: {generated_dir}")
    print(f"Output: {output_base}")
    print(f"Val split: {val_split * 100}%")
    print()
    
    # Tạo thư mục train/val
    train_dir = os.path.join(output_base, 'train')
    val_dir = os.path.join(output_base, 'val')
    
    # Xóa nếu đã tồn tại
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(val_dir):
        shutil.rmtree(val_dir)
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all classes
    classes = sorted([d for d in os.listdir(generated_dir) 
                     if os.path.isdir(os.path.join(generated_dir, d)) 
                     and not d.startswith('.')])
    
    print(f"Found {len(classes)} classes:")
    for cls in classes:
        print(f"  - {cls}")
    print()
    
    total_train = 0
    total_val = 0
    
    # Chia mỗi class thành train/val
    for class_name in classes:
        class_dir = os.path.join(generated_dir, class_name)
        
        # Get all images
        images = [f for f in os.listdir(class_dir) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if len(images) == 0:
            print(f"⚠️  WARNING: No images found in {class_name}")
            continue
        
        # Split train/val
        if len(images) < 2:
            train_images = images
            val_images = []
        else:
            train_images, val_images = train_test_split(
                images, test_size=val_split, random_state=random_state
            )
        
        # Create class directories
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Copy train images
        for img in train_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(train_class_dir, img)
            shutil.copy2(src, dst)
        
        # Copy val images
        for img in val_images:
            src = os.path.join(class_dir, img)
            dst = os.path.join(val_class_dir, img)
            shutil.copy2(src, dst)
        
        total_train += len(train_images)
        total_val += len(val_images)
        
        print(f"{class_name}:")
        print(f"  Total: {len(images)} | Train: {len(train_images)} | Val: {len(val_images)}")
    
    print()
    print(f"{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total Train: {total_train} images")
    print(f"Total Val: {total_val} images")
    print(f"Train dir: {train_dir}")
    print(f"Val dir: {val_dir}")
    print()
    
    return train_dir, val_dir

def main():
    parser = argparse.ArgumentParser(description='Prepare dataset and train ViT_Tiny')
    parser.add_argument('--generated_dir', type=str, default='/workspace/generated_images',
                       help='Directory containing generated images')
    parser.add_argument('--original_dir', type=str, 
                       default='/workspace/dataset_original/PlantVillage',
                       help='Directory containing original test images')
    parser.add_argument('--split_output', type=str, default='/workspace/data_splits_generated',
                       help='Output directory for train/val splits')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio (default: 0.2)')
    parser.add_argument('--skip_split', action='store_true',
                       help='Skip dataset splitting (use existing splits)')
    
    # Training args
    parser.add_argument('--exp_name', type=str, default='vit_tiny_generated',
                       help='Experiment name')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--pretrained', action='store_true',
                       help='Use pretrained weights')
    parser.add_argument('--output_dir', type=str, default='/workspace/experiments_results',
                       help='Output directory for results')
    parser.add_argument('--force_cpu', action='store_true',
                       help='Force CPU training (for CUDA compatibility issues)')
    
    args = parser.parse_args()
    
    # Step 1: Prepare dataset splits
    if not args.skip_split:
        train_dir, val_dir = prepare_dataset_splits(
            args.generated_dir,
            args.split_output,
            args.val_split
        )
    else:
        train_dir = os.path.join(args.split_output, 'train')
        val_dir = os.path.join(args.split_output, 'val')
        print(f"Using existing splits:")
        print(f"  Train: {train_dir}")
        print(f"  Val: {val_dir}")
        print()
    
    # Step 2: Run training
    print("="*80)
    print("STARTING TRAINING")
    print("="*80)
    print()
    
    # Build command for train_classifier.py
    cmd = [
        'python3', '/workspace/scripts/train_classifier.py',
        '--train_dir', train_dir,
        '--val_dir', val_dir,
        '--test_dir', args.original_dir,
        '--model', 'vit_tiny',
        '--exp_name', args.exp_name,
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--output_dir', args.output_dir,
        '--num_classes', '15',
        '--optimizer', 'adam',
        '--scheduler', 'cosine'
    ]
    
    if args.pretrained:
        cmd.append('--pretrained')
    
    if args.force_cpu:
        cmd.append('--force_cpu')
    
    # Execute
    import subprocess
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ Training completed successfully!")
    else:
        print(f"\n✗ Training failed with code {result.returncode}")
        return result.returncode
    
    return 0

if __name__ == '__main__':
    exit(main())
