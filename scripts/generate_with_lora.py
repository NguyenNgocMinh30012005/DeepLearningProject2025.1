#!/usr/bin/env python3
"""
Generate synthetic images using LoRA weights + ControlNet for PlantVillage dataset
1024 images per class, 15 classes total = 15,360 images
"""

import argparse
import json
import random
from pathlib import Path
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from PIL import Image, ImageFilter
import cv2
import numpy as np
from tqdm import tqdm

# Mapping: 15 classes to 15 LoRA weights
CLASS_TO_LORA = {
    'Pepper__bell___Bacterial_spot': 'pytorch_lora_weights.safetensors',
    'Pepper__bell___healthy': 'pytorch_lora_weights (1).safetensors',
    'Potato___Early_blight': 'pytorch_lora_weights (2).safetensors',
    'Potato___Late_blight': 'pytorch_lora_weights (3).safetensors',
    'Potato___healthy': 'pytorch_lora_weights (4).safetensors',
    'Tomato_Bacterial_spot': 'pytorch_lora_weights (5).safetensors',
    'Tomato_Early_blight': 'pytorch_lora_weights (6).safetensors',
    'Tomato_Late_blight': 'pytorch_lora_weights (7).safetensors',
    'Tomato_Leaf_Mold': 'pytorch_lora_weights (8).safetensors',
    'Tomato_Septoria_leaf_spot': 'pytorch_lora_weights (9).safetensors',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'pytorch_lora_weights (10).safetensors',
    'Tomato__Target_Spot': 'pytorch_lora_weights (11).safetensors',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'pytorch_lora_weights (12).safetensors',
    'Tomato__Tomato_mosaic_virus': 'pytorch_lora_weights (13).safetensors',
    'Tomato_healthy': 'pytorch_lora_weights (14).safetensors',
}

# Natural language prompts
CLASS_TO_PROMPT = {
    'Pepper__bell___Bacterial_spot': 'bell pepper leaf with bacterial spot disease',
    'Pepper__bell___healthy': 'healthy bell pepper leaf',
    'Potato___Early_blight': 'potato leaf with early blight disease',
    'Potato___Late_blight': 'potato leaf with late blight disease',
    'Potato___healthy': 'healthy potato leaf',
    'Tomato_Bacterial_spot': 'tomato leaf with bacterial spot disease',
    'Tomato_Early_blight': 'tomato leaf with early blight disease',
    'Tomato_Late_blight': 'tomato leaf with late blight disease',
    'Tomato_Leaf_Mold': 'tomato leaf with leaf mold disease',
    'Tomato_Septoria_leaf_spot': 'tomato leaf with septoria leaf spot disease',
    'Tomato_Spider_mites_Two_spotted_spider_mite': 'tomato leaf with spider mites infestation',
    'Tomato__Target_Spot': 'tomato leaf with target spot disease',
    'Tomato__Tomato_YellowLeaf__Curl_Virus': 'tomato leaf with yellow leaf curl virus disease',
    'Tomato__Tomato_mosaic_virus': 'tomato leaf with tomato mosaic virus disease',
    'Tomato_healthy': 'healthy tomato leaf',
}


def get_canny_edge(image_path, low_threshold=50, high_threshold=150):
    """Generate Canny edge map from image"""
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    return Image.fromarray(edges)


def create_random_edge_map(height=512, width=512):
    """Create a random edge map for guidance"""
    # Create random noise
    noise = np.random.randint(0, 255, (height, width), dtype=np.uint8)
    
    # Apply blur
    noise = cv2.GaussianBlur(noise, (5, 5), 0)
    
    # Apply Canny
    edges = cv2.Canny(noise, 50, 150)
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    return Image.fromarray(edges)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with LoRA + ControlNet")
    parser.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="Base Stable Diffusion model")
    parser.add_argument("--controlnet_model", type=str, default="lllyasviel/sd-controlnet-canny",
                       help="ControlNet model for Canny edges (SD1.5)")
    parser.add_argument("--lora_dir", type=str, default="/workspace/LoRA_W",
                       help="Directory containing LoRA weights")
    parser.add_argument("--out_dir", type=str, default="/workspace/generated_images",
                       help="Output directory for generated images")
    parser.add_argument("--n_per_class", type=int, default=1024,
                       help="Number of images to generate per class")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size for generation")
    parser.add_argument("--height", type=int, default=512,
                       help="Image height")
    parser.add_argument("--width", type=int, default=512,
                       help="Image width")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                       help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Guidance scale")
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=0.5,
                       help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed base")
    parser.add_argument("--use_random_edges", action='store_true',
                       help="Use random edge maps instead of from dataset")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("PlantVillage Image Generation with LoRA + ControlNet")
    print("=" * 80)
    print(f"Base model: {args.base_model}")
    print(f"ControlNet: {args.controlnet_model}")
    print(f"LoRA directory: {args.lora_dir}")
    print(f"Images per class: {args.n_per_class}")
    print(f"Total images: {args.n_per_class * len(CLASS_TO_LORA)}")
    print(f"Output: {args.out_dir}")
    print()
    
    # Load ControlNet
    print("Loading ControlNet...")
    controlnet = ControlNetModel.from_pretrained(
        args.controlnet_model,
        torch_dtype=torch.float16
    )
    
    # Load base pipeline
    print("Loading Stable Diffusion pipeline...")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base_model,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    
    # Optimize
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    
    # Try xformers
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("✓ xformers enabled")
    except:
        print("⚠️  xformers not available")
    
    print("✓ Pipeline ready!\n")
    
    # Generate for each class
    for class_idx, (class_name, lora_file) in enumerate(CLASS_TO_LORA.items()):
        print(f"\n[{class_idx + 1}/15] {class_name}")
        print("=" * 80)
        
        # Load LoRA weights for this class
        lora_path = Path(args.lora_dir) / lora_file
        if not lora_path.exists():
            print(f"⚠️  LoRA weights not found: {lora_path}, skipping...")
            continue
        
        print(f"Loading LoRA: {lora_file}")
        pipe.load_lora_weights(str(lora_path))
        
        # Create output directory
        class_out_dir = Path(args.out_dir) / class_name
        class_out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get prompt
        natural_prompt = CLASS_TO_PROMPT[class_name]
        prompt = f"a high-resolution photo of {natural_prompt}, macro photography, sharp focus, even lighting, laboratory style, plain background, professional image, detailed leaf texture"
        negative_prompt = "blurry, low quality, distorted, watermark, text, multiple leaves, background clutter, poor lighting, artifacts, people, animals, buildings"
        
        print(f"Prompt: '{prompt[:70]}...'")
        print(f"Generating {args.n_per_class} images...")
        
        # Generate images
        n_generated = 0
        batch_idx = 0
        
        pbar = tqdm(total=args.n_per_class, desc=class_name)
        
        while n_generated < args.n_per_class:
            batch_size = min(args.batch_size, args.n_per_class - n_generated)
            
            # Generate or load edge maps
            edge_maps = []
            for i in range(batch_size):
                if args.use_random_edges:
                    edge = create_random_edge_map(args.height, args.width)
                else:
                    # Create simple edge map (you can customize this)
                    edge = create_random_edge_map(args.height, args.width)
                edge_maps.append(edge)
            
            # Generate
            generators = [
                torch.Generator(device="cuda").manual_seed(args.seed + batch_idx * 1000 + i)
                for i in range(batch_size)
            ]
            
            images = pipe(
                prompt=[prompt] * batch_size,
                negative_prompt=[negative_prompt] * batch_size,
                image=edge_maps,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_conditioning_scale,
                generator=generators,
            ).images
            
            # Save
            for i, img in enumerate(images):
                seed = args.seed + batch_idx * 1000 + i
                img_name = f"{class_name}_{n_generated:05d}_seed{seed}.jpg"
                img.save(class_out_dir / img_name, quality=95)
                n_generated += 1
                pbar.update(1)
            
            batch_idx += 1
        
        pbar.close()
        print(f"✓ {n_generated} images generated for {class_name}")
    
    # Save metadata
    metadata = {
        "base_model": args.base_model,
        "controlnet_model": args.controlnet_model,
        "lora_directory": args.lora_dir,
        "n_per_class": args.n_per_class,
        "total_images": args.n_per_class * len(CLASS_TO_LORA),
        "batch_size": args.batch_size,
        "image_size": f"{args.height}x{args.width}",
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "controlnet_conditioning_scale": args.controlnet_conditioning_scale,
        "seed": args.seed,
        "class_to_lora": CLASS_TO_LORA,
    }
    
    metadata_path = Path(args.out_dir) / "generation_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✓ GENERATION COMPLETE!")
    print(f"Total classes: {len(CLASS_TO_LORA)}")
    print(f"Images per class: {args.n_per_class}")
    print(f"Total images: {args.n_per_class * len(CLASS_TO_LORA)}")
    print(f"Output directory: {args.out_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
