#!/usr/bin/env python3
"""
Tạo tổng hợp tất cả confusion matrices
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import numpy as np

# Experiments to visualize
experiments = [
    ('baseline_resnet18_real', 'Baseline: ResNet18 (Train Real)'),
    ('exp1_resnet18_syn', 'Exp1.1: ResNet18 (Train Syn, Test Real)'),
    ('exp1_resnet50_syn', 'Exp1.2: ResNet50 (Train Syn, Test Real)'),
    ('exp2_resnet18_balanced', 'Exp2: ResNet18 (Train Balanced, Test Real)'),
]

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(20, 20))
axes = axes.flatten()

results_dir = Path('/workspace/experiments_results')

for idx, (exp_dir, title) in enumerate(experiments):
    cm_path = results_dir / exp_dir / 'confusion_matrix.png'
    
    if cm_path.exists():
        img = mpimg.imread(str(cm_path))
        axes[idx].imshow(img)
        axes[idx].set_title(title, fontsize=14, fontweight='bold', pad=10)
        axes[idx].axis('off')
    else:
        axes[idx].text(0.5, 0.5, f'Not found:\n{cm_path}', 
                      ha='center', va='center', fontsize=12)
        axes[idx].set_title(title, fontsize=14, fontweight='bold')
        axes[idx].axis('off')

plt.tight_layout()

# Save
output_path = results_dir / 'visualizations' / 'all_confusion_matrices.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✅ Saved: {output_path}")

# Also create individual copies in visualizations folder
for exp_dir, title in experiments:
    src = results_dir / exp_dir / 'confusion_matrix.png'
    dst = results_dir / 'visualizations' / f'{exp_dir}_confusion_matrix.png'
    if src.exists():
        import shutil
        shutil.copy2(src, dst)
        print(f"✅ Copied: {dst}")

plt.close()

# List all confusion matrices
print("\n" + "="*80)
print("CONFUSION MATRICES:")
print("="*80)
for exp_dir, title in experiments:
    cm_path = results_dir / exp_dir / 'confusion_matrix.png'
    if cm_path.exists():
        print(f"✅ {title}")
        print(f"   Path: {cm_path}")
    else:
        print(f"❌ {title}")
        print(f"   Missing: {cm_path}")
