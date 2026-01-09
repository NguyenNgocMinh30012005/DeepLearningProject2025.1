#!/usr/bin/env python3
"""
Save all configurations and hyperparameters
"""
import json
import os
from datetime import datetime

def main():
    config = {
        "experiment_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiments": {
            "exp1": {
                "name": "Train Synthetic, Test Real",
                "description": "Train và validate trên synthetic data, test trên real data",
                "models": ["ResNet18", "ResNet50"],
                "train_data": "Synthetic (15,360 images, 1024/class)",
                "val_data": "Synthetic (15,360 images, 1024/class)",
                "test_data": "Real (Test split)"
            },
            "exp2": {
                "name": "Train Balanced (Real+Syn), Test Real",
                "description": "Train trên balanced dataset (real + synthetic để cân bằng imbalance), test trên real",
                "models": ["ResNet18"],
                "train_data": "Balanced (Real + Synthetic, target 1024/class)",
                "val_data": "Balanced (Real + Synthetic, target 256/class)",
                "test_data": "Real (Test split)"
            },
            "baseline": {
                "name": "Train Real Only (Baseline)",
                "description": "Baseline - Train chỉ trên real data (imbalanced)",
                "models": ["ResNet18"],
                "train_data": "Real (Train split, imbalanced)",
                "val_data": "Real (Val split)",
                "test_data": "Real (Test split)"
            }
        },
        "hyperparameters": {
            "common": {
                "epochs": 50,
                "batch_size": 32,
                "learning_rate": 0.001,
                "optimizer": "Adam",
                "scheduler": "CosineAnnealingLR",
                "weight_decay": 1e-4,
                "loss_function": "CrossEntropyLoss",
                "data_augmentation": [
                    "Resize(224x224)",
                    "RandomHorizontalFlip",
                    "RandomRotation(10°)",
                    "ColorJitter(brightness=0.2, contrast=0.2)",
                    "Normalize(ImageNet mean/std)"
                ]
            },
            "resnet18": {
                "architecture": "ResNet18",
                "pretrained": True,
                "pretrained_on": "ImageNet",
                "parameters": "~11.7M",
                "modified_layers": "Final FC layer (512 -> 15 classes)"
            },
            "resnet50": {
                "architecture": "ResNet50",
                "pretrained": True,
                "pretrained_on": "ImageNet",
                "parameters": "~25.6M",
                "modified_layers": "Final FC layer (2048 -> 15 classes)"
            },
            "vit_tiny": {
                "architecture": "Vision Transformer Tiny",
                "patch_size": "16x16",
                "image_size": "224x224",
                "pretrained": True,
                "pretrained_on": "ImageNet",
                "parameters": "~5.7M",
                "modified_layers": "Head (192 -> 15 classes)"
            }
        },
        "dataset_info": {
            "name": "PlantVillage",
            "num_classes": 15,
            "classes": [
                "Pepper__bell___Bacterial_spot",
                "Pepper__bell___healthy",
                "Potato___Early_blight",
                "Potato___Late_blight",
                "Potato___healthy",
                "Tomato_Bacterial_spot",
                "Tomato_Early_blight",
                "Tomato_Late_blight",
                "Tomato_Leaf_Mold",
                "Tomato_Septoria_leaf_spot",
                "Tomato_Spider_mites_Two_spotted_spider_mite",
                "Tomato__Target_Spot",
                "Tomato__Tomato_YellowLeaf__Curl_Virus",
                "Tomato__Tomato_mosaic_virus",
                "Tomato_healthy"
            ],
            "real_dataset": {
                "total_images": 20638,
                "imbalanced": True,
                "min_class_size": 152,
                "max_class_size": 3209,
                "split_ratio": {
                    "train": 0.7,
                    "val": 0.15,
                    "test": 0.15
                }
            },
            "synthetic_dataset": {
                "total_images": 15360,
                "images_per_class": 1024,
                "generation_method": "LoRA finetuned + ControlNet Canny",
                "base_model": "runwayml/stable-diffusion-v1-5",
                "controlnet": "lllyasviel/sd-controlnet-canny",
                "fid_score": 31.00,
                "inception_score": 1.05
            }
        },
        "metrics": {
            "training": ["Loss", "Accuracy"],
            "validation": ["Loss", "Accuracy"],
            "test": [
                "Accuracy",
                "Precision (weighted)",
                "Recall (weighted)",
                "F1 Score (weighted)",
                "Per-class Precision",
                "Per-class Recall",
                "Per-class F1",
                "Confusion Matrix"
            ]
        },
        "hardware": {
            "device": "CUDA (GPU)" if os.popen("nvidia-smi").read() else "CPU",
            "framework": "PyTorch",
            "libraries": [
                "torch",
                "torchvision",
                "timm",
                "scikit-learn",
                "matplotlib",
                "seaborn"
            ]
        }
    }
    
    # Save config
    output_dir = "/workspace/experiments_results"
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, "experiments_config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"{'='*80}")
    print("EXPERIMENT CONFIGURATION")
    print(f"{'='*80}")
    print(json.dumps(config, indent=2))
    print(f"\n✓ Config saved to: {config_path}")

if __name__ == '__main__':
    main()
