#!/usr/bin/env python3
"""
Prepare PlantVillage dataset for DataDream format
DataDream expects: data/$DATASET/real_train_fewshot/seed$SEED/$CLASS_NAME/$FILE
"""

import os
import shutil
from pathlib import Path
import random
from collections import defaultdict

# Paths
PLANTVILLAGE_DIR = Path("/workspace/dataset_original/PlantVillage")
OUTPUT_DIR = Path("/workspace/DataDream-1/data/plantvillage")
N_SHOT = 16
SEED = 0

def prepare_fewshot_data():
    """Prepare few-shot data in DataDream format"""
    
    print("="*80)
    print("PREPARING PLANTVILLAGE DATA FOR DATADREAM")
    print("="*80)
    
    # Set seed for reproducibility
    random.seed(SEED)
    
    # Create output directory
    fewshot_dir = OUTPUT_DIR / "real_train_fewshot" / f"seed{SEED}"
    fewshot_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSource: {PLANTVILLAGE_DIR}")
    print(f"Target: {fewshot_dir}")
    print(f"N-shot: {N_SHOT}")
    
    # Get all classes
    classes = sorted([d.name for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir()])
    print(f"\nFound {len(classes)} classes")
    
    stats = defaultdict(int)
    
    for class_idx, class_name in enumerate(classes):
        print(f"\n[{class_idx+1}/{len(classes)}] {class_name}")
        
        # Get all images in this class
        class_dir = PLANTVILLAGE_DIR / class_name
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
            image_files.extend(list(class_dir.glob(f'*{ext}')))
        
        print(f"  Total images: {len(image_files)}")
        
        if len(image_files) < N_SHOT:
            print(f"  ⚠️  WARNING: Only {len(image_files)} images, need {N_SHOT}")
            selected = image_files
        else:
            # Randomly sample N_SHOT images
            selected = random.sample(image_files, N_SHOT)
        
        # Create class directory
        target_class_dir = fewshot_dir / class_name
        target_class_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy selected images
        for img_file in selected:
            target_file = target_class_dir / img_file.name
            shutil.copy2(img_file, target_file)
        
        print(f"  ✓ Copied {len(selected)} images")
        stats[class_name] = len(selected)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal classes: {len(stats)}")
    print(f"Total images: {sum(stats.values())}")
    print(f"Average per class: {sum(stats.values())/len(stats):.1f}")
    print(f"\nMin: {min(stats.values())} images")
    print(f"Max: {max(stats.values())} images")
    
    print(f"\n✓ Few-shot data prepared at: {fewshot_dir}")
    
    return fewshot_dir, classes

def create_class_list(classes):
    """Create class list file for DataDream"""
    
    # Save class list
    class_list_file = OUTPUT_DIR / "class_names.txt"
    with open(class_list_file, 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    
    print(f"\n✓ Class list saved to: {class_list_file}")
    
    # Also save to util_data.py format
    util_data_snippet = OUTPUT_DIR / "util_data_snippet.py"
    with open(util_data_snippet, 'w') as f:
        f.write('SUBSET_NAMES = {\n')
        f.write('    "plantvillage": [\n')
        for class_name in classes:
            f.write(f'        "{class_name}",\n')
        f.write('    ]\n')
        f.write('}\n')
    
    print(f"✓ util_data snippet saved to: {util_data_snippet}")
    print("\nYou can copy this to DataDream/util_data.py")

def prepare_test_data():
    """Copy full test data"""
    
    print("\n" + "="*80)
    print("PREPARING TEST DATA")
    print("="*80)
    
    test_dir = OUTPUT_DIR / "real_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nTarget: {test_dir}")
    
    # Just create symlinks to save space
    classes = sorted([d.name for d in PLANTVILLAGE_DIR.iterdir() if d.is_dir()])
    
    for class_name in classes:
        source = PLANTVILLAGE_DIR / class_name
        target = test_dir / class_name
        
        if target.exists():
            target.unlink()
        
        target.symlink_to(source, target_is_directory=True)
    
    print(f"\n✓ Test data symlinked to: {test_dir}")

if __name__ == "__main__":
    # Prepare few-shot data
    fewshot_dir, classes = prepare_fewshot_data()
    
    # Create class list
    create_class_list(classes)
    
    # Prepare test data
    prepare_test_data()
    
    print("\n" + "="*80)
    print("✅ ALL DATA PREPARED FOR DATADREAM!")
    print("="*80)
    print("\nNext steps:")
    print("1. Update DataDream-1/DataDream/local.yaml")
    print("2. Add plantvillage to util_data.py")
    print("3. Run DataDream training")
