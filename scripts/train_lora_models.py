#!/usr/bin/env python3
"""
Retrain LoRA cho 5 failed Tomato classes
Overwrite old weights và generated images
"""
import os
import sys
import json
import random
import shutil
import subprocess
from pathlib import Path
from PIL import Image

# Failed classes cần retrain
FAILED_CLASSES = [
    'Tomato_Bacterial_spot',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato__Target_Spot',
    'Tomato_healthy',
]

# Training config (từ DataDream notebook)
BASE_MODEL = "friedrichor/stable-diffusion-2-1-realistic"
RESOLUTION = 512
TRAIN_BATCH = 1
GRAD_ACCUM = 1
LR = 1e-4
RANK = 16
MAX_TRAIN_STEPS = 1200  # Tăng từ 800 -> 1200 để học tốt hơn
SAVE_STEPS = 300
N_SHOT = 16  # Few-shot: 16 images per class
SEED = 42

# Directories
REAL_DATA_DIR = Path("/workspace/dataset_original/PlantVillage")
FEWSHOT_DIR = Path("/workspace/fewshot_retrain")
LORA_OUTPUT_DIR = Path("/workspace/LoRA_W")  # Overwrite old weights
GENERATED_DIR = Path("/workspace/generated_images")  # Overwrite old images

# Improved prompts với more details
IMPROVED_PROMPTS = {
    'Tomato_Bacterial_spot': (
        'a high quality macro photograph of sks tomato leaf with bacterial spot disease, '
        'showing dark brown circular lesions with yellow halos, detailed plant pathology, '
        'realistic disease symptoms, sharp focus, natural lighting'
    ),
    'Tomato_Late_blight': (
        'a high quality macro photograph of sks tomato leaf with late blight disease, '
        'showing large irregular dark brown necrotic lesions, water-soaked appearance, '
        'detailed plant pathology, realistic disease symptoms, sharp focus, natural lighting'
    ),
    'Tomato_Leaf_Mold': (
        'a high quality macro photograph of sks tomato leaf with leaf mold disease, '
        'showing yellow patches on upper surface and olive-green mold on lower surface, '
        'detailed plant pathology, realistic disease symptoms, sharp focus, natural lighting'
    ),
    'Tomato__Target_Spot': (
        'a high quality macro photograph of sks tomato leaf with target spot disease, '
        'showing concentric rings pattern like target, brown lesions with lighter center, '
        'detailed plant pathology, realistic disease symptoms, sharp focus, natural lighting'
    ),
    'Tomato_healthy': (
        'a high quality macro photograph of sks healthy tomato leaf, '
        'vibrant green color, no disease symptoms, clean and fresh appearance, '
        'natural plant, sharp focus, natural lighting'
    ),
}

# LoRA weight mapping
CLASS_TO_LORA_FILE = {
    'Tomato_Bacterial_spot': 'pytorch_lora_weights (5).safetensors',
    'Tomato_Late_blight': 'pytorch_lora_weights (7).safetensors',
    'Tomato_Leaf_Mold': 'pytorch_lora_weights (8).safetensors',
    'Tomato__Target_Spot': 'pytorch_lora_weights (11).safetensors',
    'Tomato_healthy': 'pytorch_lora_weights (14).safetensors',
}


def prepare_fewshot_data():
    """Chuẩn bị few-shot data cho 5 classes"""
    print("="*80)
    print("STEP 1: Preparing Few-Shot Data")
    print("="*80)
    
    FEWSHOT_DIR.mkdir(parents=True, exist_ok=True)
    
    for class_name in FAILED_CLASSES:
        class_dir = REAL_DATA_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️  Class directory not found: {class_dir}")
            continue
        
        # Get all images
        images = [f for f in class_dir.iterdir() 
                 if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
        
        if len(images) < N_SHOT:
            print(f"⚠️  {class_name}: Only {len(images)} images (need {N_SHOT})")
            n_shot = len(images)
        else:
            n_shot = N_SHOT
        
        # Sample few-shot
        random.seed(SEED)
        sampled = random.sample(images, n_shot)
        
        # Copy to fewshot dir
        out_dir = FEWSHOT_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in sampled:
            shutil.copy2(img_path, out_dir / img_path.name)
        
        print(f"✓ {class_name}: {n_shot} images -> {out_dir}")
    
    print(f"\n✓ Few-shot data ready: {FEWSHOT_DIR}")


def train_lora_for_class(class_name, gpu_id=0):
    """Train LoRA cho một class"""
    print(f"\n{'='*80}")
    print(f"Training LoRA: {class_name}")
    print(f"{'='*80}")
    
    instance_dir = FEWSHOT_DIR / class_name
    if not instance_dir.exists():
        print(f"⚠️  Fewshot dir not found: {instance_dir}")
        return False
    
    # Output dir cho LoRA weights
    lora_class_dir = LORA_OUTPUT_DIR / class_name
    lora_class_dir.mkdir(parents=True, exist_ok=True)
    
    # Improved prompt
    instance_prompt = IMPROVED_PROMPTS[class_name]
    
    # Training command
    cmd = [
        "accelerate", "launch",
        "--num_processes", "1",
        "--mixed_precision", "no",  # Changed from fp16 to no (fp32)
        "--main_process_port", "29500",
        "/workspace/diffusers_repo/examples/dreambooth/train_dreambooth_lora.py",
        "--pretrained_model_name_or_path", BASE_MODEL,
        "--instance_data_dir", str(instance_dir),
        "--output_dir", str(lora_class_dir),
        "--instance_prompt", instance_prompt,
        "--resolution", str(RESOLUTION),
        "--train_batch_size", str(TRAIN_BATCH),
        "--gradient_accumulation_steps", str(GRAD_ACCUM),
        "--learning_rate", str(LR),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(MAX_TRAIN_STEPS),
        "--checkpointing_steps", str(SAVE_STEPS),
        "--seed", str(SEED),
        "--rank", str(RANK),
        "--train_text_encoder",
        "--gradient_checkpointing",
    ]
    
    # Set GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    print(f"\nCommand: {' '.join(cmd[:10])}...")
    print(f"Instance prompt: {instance_prompt[:100]}...")
    print(f"Output: {lora_class_dir}")
    print(f"\nStarting training (MAX_TRAIN_STEPS={MAX_TRAIN_STEPS})...\n")
    
    try:
        result = subprocess.run(cmd, check=True, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True)
        
        # Print last 20 lines
        lines = result.stdout.split('\n')
        print('\n'.join(lines[-20:]))
        
        # Copy weight file to main LoRA_W directory với tên chuẩn
        weight_file = lora_class_dir / "pytorch_lora_weights.safetensors"
        if weight_file.exists():
            target_file = LORA_OUTPUT_DIR / CLASS_TO_LORA_FILE[class_name]
            shutil.copy2(weight_file, target_file)
            print(f"\n✓ Weight copied: {target_file.name}")
            return True
        else:
            print(f"\n⚠️  Weight file not found: {weight_file}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed for {class_name}")
        print(f"Error: {e}")
        return False


def generate_images_for_class(class_name):
    """Generate images cho một class (overwrite old ones)"""
    print(f"\n{'='*80}")
    print(f"Generating images: {class_name}")
    print(f"{'='*80}")
    
    # Delete old images
    class_gen_dir = GENERATED_DIR / class_name
    if class_gen_dir.exists():
        print(f"Deleting old images: {class_gen_dir}")
        shutil.rmtree(class_gen_dir)
    class_gen_dir.mkdir(parents=True, exist_ok=True)
    
    # Import libraries
    import torch
    from diffusers import StableDiffusionPipeline
    from tqdm import tqdm
    
    # Load model
    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe = pipe.to("cuda")
    
    # Load LoRA weights
    lora_file = LORA_OUTPUT_DIR / CLASS_TO_LORA_FILE[class_name]
    if not lora_file.exists():
        print(f"⚠️  LoRA weight not found: {lora_file}")
        return False
    
    print(f"Loading LoRA: {lora_file.name}")
    pipe.load_lora_weights(str(LORA_OUTPUT_DIR), weight_name=CLASS_TO_LORA_FILE[class_name])
    
    # Generate
    prompt = IMPROVED_PROMPTS[class_name]
    negative_prompt = (
        "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
        "watermark, text, signature, cartoon, painting, illustration"
    )
    
    num_images = 1024
    batch_size = 4
    num_inference_steps = 30
    guidance_scale = 7.5
    
    print(f"Generating {num_images} images...")
    print(f"Prompt: {prompt[:100]}...")
    
    for i in tqdm(range(0, num_images, batch_size)):
        batch = min(batch_size, num_images - i)
        
        # Unique seed per image
        generators = [torch.Generator(device="cuda").manual_seed(SEED + i + j) 
                     for j in range(batch)]
        
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            images = pipe(
                [prompt] * batch,
                negative_prompt=[negative_prompt] * batch,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generators,
            ).images
        
        # Save
        for j, img in enumerate(images):
            img_name = f"{class_name}_{i+j:05d}_seed{SEED+i+j}.jpg"
            img.save(class_gen_dir / img_name, quality=95, optimize=True)
    
    print(f"✓ Generated {num_images} images: {class_gen_dir}")
    
    # Cleanup
    del pipe
    torch.cuda.empty_cache()
    
    return True


def main():
    print("\n" + "="*80)
    print("RETRAIN LORA FOR FAILED CLASSES")
    print("="*80)
    print(f"\nFailed classes to retrain:")
    for i, cls in enumerate(FAILED_CLASSES, 1):
        print(f"  {i}. {cls}")
    
    print(f"\nImproved training:")
    print(f"  - MAX_TRAIN_STEPS: 800 → 1200 (+50%)")
    print(f"  - Better prompts với more disease details")
    print(f"  - Will overwrite old weights & images")
    
    # Step 1: Prepare few-shot data
    prepare_fewshot_data()
    
    # Step 2: Train LoRA for each class
    print("\n" + "="*80)
    print("STEP 2: Training LoRA")
    print("="*80)
    
    success_classes = []
    failed_train = []
    
    for class_name in FAILED_CLASSES:
        success = train_lora_for_class(class_name, gpu_id=0)
        if success:
            success_classes.append(class_name)
        else:
            failed_train.append(class_name)
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"✓ Success: {len(success_classes)}/{len(FAILED_CLASSES)}")
    if failed_train:
        print(f"❌ Failed: {failed_train}")
    
    # Step 3: Generate images
    if success_classes:
        print("\n" + "="*80)
        print("STEP 3: Generating Images")
        print("="*80)
        
        for class_name in success_classes:
            try:
                generate_images_for_class(class_name)
            except Exception as e:
                print(f"❌ Failed to generate for {class_name}: {e}")
    
    print("\n" + "="*80)
    print("✓ RETRAIN COMPLETE!")
    print("="*80)
    print(f"\nNew LoRA weights: {LORA_OUTPUT_DIR}")
    print(f"New images: {GENERATED_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Verify generated images quality")
    print(f"  2. Retrain classifier")
    print(f"  3. Compare new results with old")


if __name__ == '__main__':
    main()
