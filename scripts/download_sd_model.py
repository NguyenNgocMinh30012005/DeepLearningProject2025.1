#!/usr/bin/env python3
"""Download Stable Diffusion model for generation"""

import torch
from diffusers import StableDiffusionPipeline

print("=" * 80)
print("📥 DOWNLOADING STABLE DIFFUSION MODEL")
print("=" * 80)
print("")
print("Model: friedrichor/stable-diffusion-2-1-realistic")
print("This will take a few minutes...")
print("")

try:
    print("Loading model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "friedrichor/stable-diffusion-2-1-realistic",
        torch_dtype=torch.float16,
    )
    print("")
    print("✅ Model downloaded successfully!")
    print(f"Cached location: ~/.cache/huggingface/hub/")
    print("")
    
except Exception as e:
    print(f"❌ Error downloading model: {e}")
    print("")
    print("Please check:")
    print("  1. Internet connection")
    print("  2. HuggingFace Hub access")
    print("  3. Model name is correct")
    exit(1)
