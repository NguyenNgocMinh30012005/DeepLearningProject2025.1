#!/usr/bin/env python3
"""
Generate images using LoRA weights ONLY (no ControlNet) - Faster và đơn giản hơn
"""

import argparse
import json
from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from tqdm import tqdm

# 15 classes mapping
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="friedrichor/stable-diffusion-2-1-realistic")
    parser.add_argument("--lora_dir", type=str, default="/workspace/LoRA_W")
    parser.add_argument("--out_dir", type=str, default="/workspace/generated_images")
    parser.add_argument("--n_per_class", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("="*80)
    print("PlantVillage Image Generation with LoRA (Text-to-Image Only)")
    print("="*80)
    print(f"Model: {args.base_model}")
    print(f"LoRA dir: {args.lora_dir}")
    print(f"Output: {args.out_dir}")
    print(f"Images per class: {args.n_per_class}")
    print(f"Total: {args.n_per_class * 15} images")
    print()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load base pipeline ONCE
    print("Loading base pipeline...")
    base_pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        safety_checker=None,
    )
    base_pipe.scheduler = UniPCMultistepScheduler.from_config(base_pipe.scheduler.config)
    base_pipe = base_pipe.to(device)
    
    try:
        base_pipe.enable_xformers_memory_efficient_attention()
        print("✓ xformers enabled")
    except:
        pass
    
    print("✓ Base pipeline ready!\n")
    
    # Generate for each class
    for idx, (class_name, lora_file) in enumerate(CLASS_TO_LORA.items()):
        print(f"\n[{idx+1}/15] {class_name}")
        print("="*80)
        
        lora_path = Path(args.lora_dir) / lora_file
        if not lora_path.exists():
            print(f"⚠️  LoRA not found: {lora_path}, skipping")
            continue
        
        # Load LoRA
        print(f"Loading LoRA: {lora_file}")
        base_pipe.load_lora_weights(str(lora_path))
        
        # Create output dir
        class_dir = Path(args.out_dir) / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        # Prompt
        desc = CLASS_TO_PROMPT[class_name]
        prompt = f"a high-resolution photo of {desc}, macro photography, sharp focus, even lighting, laboratory style, plain background"
        negative = "blurry, low quality, distorted, watermark, text, multiple leaves"
        
        print(f"Generating {args.n_per_class} images...")
        
        # Generate
        n_gen = 0
        batch_idx = 0
        pbar = tqdm(total=args.n_per_class, desc=class_name)
        
        while n_gen < args.n_per_class:
            bs = min(args.batch_size, args.n_per_class - n_gen)
            
            generators = [
                torch.Generator(device=device).manual_seed(args.seed + batch_idx*1000 + i)
                for i in range(bs)
            ]
            
            images = base_pipe(
                prompt=[prompt]*bs,
                negative_prompt=[negative]*bs,
                height=args.height,
                width=args.width,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                generator=generators,
            ).images
            
            for i, img in enumerate(images):
                seed = args.seed + batch_idx*1000 + i
                img.save(class_dir / f"{class_name}_{n_gen:05d}_seed{seed}.jpg", quality=95)
                n_gen += 1
                pbar.update(1)
            
            batch_idx += 1
        
        pbar.close()
        print(f"✓ {n_gen} images generated")
    
    # Save metadata
    metadata = {
        "model": args.base_model,
        "lora_dir": args.lora_dir,
        "n_per_class": args.n_per_class,
        "total_images": args.n_per_class * 15,
        "method": "text-to-image with LoRA (no ControlNet)",
    }
    
    with open(Path(args.out_dir) / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("✓ GENERATION COMPLETE!")
    print(f"Total images: {args.n_per_class * 15}")
    print(f"Output: {args.out_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
