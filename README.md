# 🌿 Plant Disease Classification with Synthetic Data

## 📊 Project Overview

This project demonstrates **plant disease classification** using both **real and synthetic data**. We generate high-quality synthetic images using **Stable Diffusion 1.5 with LoRA fine-tuning**, then train classifiers (ResNet18, ResNet50) to evaluate synthetic data effectiveness.

---

## 📁 Project Structure

```
workspace/
├── 📊 DATA & MODELS
│   ├── dataset_original/          # Original PlantVillage dataset
│   ├── dataset_prepared/          # Train/val/test splits + balanced data
│   ├── generated_images/          # ✨ 15,360 synthetic images (1024/class)
│   ├── LoRA_W/                    # LoRA weights (15 files, 1/class)
│   └── fewshot_data/              # Few-shot training data (16 samples/class)
│
├── 🔧 SCRIPTS
│   ├── train_lora_models.py       # Train LoRA for each class
│   ├── generate_lora_only.py      # Generate synthetic images
│   ├── prepare_datasets.py        # Prepare train/val/test splits
│   ├── train_classifier.py        # Train ResNet models
│   ├── compute_fid_score.py       # Calculate FID score
│   └── compute_is_score.py        # Calculate Inception Score


### **Option 2: Step-by-Step**

#### **1. Generate Synthetic Images**

```bash
cd /workspace/scripts
python3 generate_lora_only.py \
    --n_per_class 1024 \
    --batch_size 4 \
    --output_dir ../generated_images
```

#### **2. Prepare Datasets**

```bash
python3 prepare_datasets.py \
    --real_dir ../dataset_original/PlantVillage \
    --synth_dir ../generated_images \
    --output_base ../dataset_prepared
```

#### **3. Train Classifier**

```bash
python3 train_classifier.py \
    --model resnet18 \
    --train_dir ../dataset_prepared/synthetic_train \
    --val_dir ../dataset_prepared/synthetic_val \
    --test_dir ../dataset_prepared/real_test \
    --output_dir ../experiments_results \
    --exp_name my_experiment \
    --num_classes 15 \
    --epochs 50 \
    --batch_size 32 \
    --lr 0.001
```


## 📖 Dataset

**PlantVillage Dataset (15 classes):**
- Pepper: Bacterial spot, Healthy
- Potato: Early blight, Late blight, Healthy
- Tomato: 10 disease classes + Healthy

**Class Distribution:**
- Original: 152 - 3,208 images/class (21x imbalance)
- After balancing: 1,024 - 2,245 images/class (2.2x imbalance)

**Splits:**
- Train: 70% (14,434 real images)
- Val: 15% (3,100 real images)
- Test: 15% (3,104 real images)


### **Scripts**
- `scripts/train_lora_models.py`: Train LoRA adapters
- `scripts/generate_lora_only.py`: Generate synthetic images
- `scripts/train_classifier.py`: Train classifiers
- `scripts/prepare_datasets.py`: Prepare datasets

### **Master Scripts**
- `run_full_pipeline.sh`: Run everything



