#!/usr/bin/env python3
"""
Retrain LoRA for ALL PlantVillage classes (15 classes) + regenerate images per class.

What this script does (end-to-end):
1) Prepare few-shot data: sample N_SHOT real images per class -> /workspace/fewshot_retrain/<class>/
2) Train LoRA per class using diffusers DreamBooth LoRA script (accelerate launch)
   -> saves class-specific LoRA to /workspace/LoRA_W/<class>/pytorch_lora_weights.safetensors
   -> then copies it to /workspace/LoRA_W/<mapped_file_name>.safetensors (overwrite)
3) Regenerate synthetic images per class (overwrite old class folder) -> /workspace/generated_images/<class>/

Notes:
- This is "LoRA only" (no ControlNet).
- You can tune MAX_TRAIN_STEPS, RESOLUTION, RANK, steps/guidance for generation.
"""

import os
import json
import random
import shutil
import subprocess
from pathlib import Path

# -----------------------------
# Classes (15) + LoRA filenames
# -----------------------------
CLASS_TO_LORA_FILE = {
    "Pepper__bell___Bacterial_spot": "pytorch_lora_weights.safetensors",
    "Pepper__bell___healthy": "pytorch_lora_weights (1).safetensors",
    "Potato___Early_blight": "pytorch_lora_weights (2).safetensors",
    "Potato___Late_blight": "pytorch_lora_weights (3).safetensors",
    "Potato___healthy": "pytorch_lora_weights (4).safetensors",
    "Tomato_Bacterial_spot": "pytorch_lora_weights (5).safetensors",
    "Tomato_Early_blight": "pytorch_lora_weights (6).safetensors",
    "Tomato_Late_blight": "pytorch_lora_weights (7).safetensors",
    "Tomato_Leaf_Mold": "pytorch_lora_weights (8).safetensors",
    "Tomato_Septoria_leaf_spot": "pytorch_lora_weights (9).safetensors",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "pytorch_lora_weights (10).safetensors",
    "Tomato__Target_Spot": "pytorch_lora_weights (11).safetensors",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "pytorch_lora_weights (12).safetensors",
    "Tomato__Tomato_mosaic_virus": "pytorch_lora_weights (13).safetensors",
    "Tomato_healthy": "pytorch_lora_weights (14).safetensors",
}

ALL_CLASSES = list(CLASS_TO_LORA_FILE.keys())

# -----------------------------
# Training config
# -----------------------------
BASE_MODEL = "friedrichor/stable-diffusion-2-1-realistic"
RESOLUTION = 256
TRAIN_BATCH = 1
GRAD_ACCUM = 1
LR = 1e-4
RANK = 16
MAX_TRAIN_STEPS = 1200
SAVE_STEPS = 300
N_SHOT = 16
SEED = 42

# -----------------------------
# Generation config (per class)
# -----------------------------
GEN_NUM_IMAGES = 1024
GEN_BATCH_SIZE = 4
GEN_STEPS = 30
GEN_GUIDANCE = 7.5

# -----------------------------
# Directories
# -----------------------------
REAL_DATA_DIR = Path("/workspace/dataset_original/PlantVillage")
FEWSHOT_DIR = Path("/workspace/fewshot_retrain")
LORA_OUTPUT_DIR = Path("/workspace/LoRA_W")          # will overwrite mapped files
GENERATED_DIR = Path("/workspace/generated_images")  # will overwrite class folders

# -----------------------------
# Prompts
# - Provide a base description per class (for consistency)
# - Provide improved prompts for difficult classes (optional)
# -----------------------------
BASE_CLASS_DESC = {
    "Pepper__bell___Bacterial_spot": "bell pepper leaf with bacterial spot disease",
    "Pepper__bell___healthy": "healthy bell pepper leaf",
    "Potato___Early_blight": "potato leaf with early blight disease",
    "Potato___Late_blight": "potato leaf with late blight disease",
    "Potato___healthy": "healthy potato leaf",
    "Tomato_Bacterial_spot": "tomato leaf with bacterial spot disease",
    "Tomato_Early_blight": "tomato leaf with early blight disease",
    "Tomato_Late_blight": "tomato leaf with late blight disease",
    "Tomato_Leaf_Mold": "tomato leaf with leaf mold disease",
    "Tomato_Septoria_leaf_spot": "tomato leaf with septoria leaf spot disease",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "tomato leaf with spider mites infestation",
    "Tomato__Target_Spot": "tomato leaf with target spot disease",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "tomato leaf with yellow leaf curl virus disease",
    "Tomato__Tomato_mosaic_virus": "tomato leaf with tomato mosaic virus disease",
    "Tomato_healthy": "healthy tomato leaf",
}

# Optional: more detailed prompts for classes you found problematic
IMPROVED_PROMPTS = {
    "Tomato_Bacterial_spot": (
        "a high quality macro photograph of sks tomato leaf with bacterial spot disease, "
        "showing dark brown circular lesions with yellow halos, detailed plant pathology, "
        "realistic disease symptoms, sharp focus, natural lighting"
    ),
    "Tomato_Late_blight": (
        "a high quality macro photograph of sks tomato leaf with late blight disease, "
        "showing large irregular dark brown necrotic lesions, water-soaked appearance, "
        "detailed plant pathology, realistic disease symptoms, sharp focus, natural lighting"
    ),
    "Tomato_Leaf_Mold": (
        "a high quality macro photograph of sks tomato leaf with leaf mold disease, "
        "showing yellow patches on upper surface and olive-green mold on lower surface, "
        "detailed plant pathology, realistic disease symptoms, sharp focus, natural lighting"
    ),
    "Tomato__Target_Spot": (
        "a high quality macro photograph of sks tomato leaf with target spot disease, "
        "showing concentric rings pattern like target, brown lesions with lighter center, "
        "detailed plant pathology, realistic disease symptoms, sharp focus, natural lighting"
    ),
    "Tomato_healthy": (
        "a high quality macro photograph of sks healthy tomato leaf, "
        "vibrant green color, no disease symptoms, clean and fresh appearance, "
        "natural plant, sharp focus, natural lighting"
    ),
}

def build_training_prompt(class_name: str) -> str:
    """Prompt used for DreamBooth LoRA training."""
    if class_name in IMPROVED_PROMPTS:
        return IMPROVED_PROMPTS[class_name]
    desc = BASE_CLASS_DESC.get(class_name, class_name.replace("_", " "))
    return (
        f"a high quality macro photograph of sks {desc}, "
        "realistic plant pathology details, sharp focus, natural lighting"
    )

def build_generation_prompt(class_name: str) -> str:
    """Prompt used for image generation."""
    # Reuse training prompt style (works well for PlantVillage-like look)
    desc = BASE_CLASS_DESC.get(class_name, class_name.replace("_", " "))
    return (
        f"a high-resolution photo of {desc}, macro photography, sharp focus, "
        "even lighting, laboratory style, plain background"
    )

NEGATIVE_PROMPT = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, "
    "watermark, text, signature, cartoon, painting, illustration, multiple leaves"
)

# ---------------------------------------------------------
# Step 1: Prepare few-shot data for ALL classes
# ---------------------------------------------------------
def prepare_fewshot_data():
    print("=" * 80)
    print("STEP 1: Preparing Few-Shot Data (ALL classes)")
    print("=" * 80)

    FEWSHOT_DIR.mkdir(parents=True, exist_ok=True)

    # Optional: clear old fewshot data to avoid mixing old/new
    # Comment out if you want to keep previous samples
    if FEWSHOT_DIR.exists():
        for cls in ALL_CLASSES:
            cls_dir = FEWSHOT_DIR / cls
            if cls_dir.exists():
                shutil.rmtree(cls_dir)

    for class_name in ALL_CLASSES:
        class_dir = REAL_DATA_DIR / class_name
        if not class_dir.exists():
            print(f"⚠️  Class directory not found: {class_dir} (skip)")
            continue

        images = [f for f in class_dir.iterdir() if f.suffix.lower() in [".jpg", ".jpeg", ".png"]]
        if len(images) == 0:
            print(f"⚠️  {class_name}: No images found (skip)")
            continue

        n_shot = min(N_SHOT, len(images))

        # deterministic sampling per class
        rng = random.Random(SEED)  # stable across runs
        sampled = rng.sample(images, n_shot)

        out_dir = FEWSHOT_DIR / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sampled:
            shutil.copy2(img_path, out_dir / img_path.name)

        print(f"✓ {class_name}: {n_shot} images -> {out_dir}")

    print(f"\n✓ Few-shot data ready: {FEWSHOT_DIR}\n")


# ---------------------------------------------------------
# Step 2: Train LoRA for a class
# ---------------------------------------------------------
def train_lora_for_class(class_name: str, gpu_id: int = 0) -> bool:
    print(f"\n{'=' * 80}")
    print(f"Training LoRA: {class_name}")
    print(f"{'=' * 80}")

    instance_dir = FEWSHOT_DIR / class_name
    if not instance_dir.exists():
        print(f"⚠️  Fewshot dir not found: {instance_dir}")
        return False

    lora_class_dir = LORA_OUTPUT_DIR / class_name
    lora_class_dir.mkdir(parents=True, exist_ok=True)

    instance_prompt = build_training_prompt(class_name)

    cmd = [
        "accelerate", "launch",
        "--num_processes", "1",
        "--mixed_precision", "no",  # FP32 for stability
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Instance prompt: {instance_prompt[:120]}...")
    print(f"Output dir: {lora_class_dir}")
    print(f"MAX_TRAIN_STEPS={MAX_TRAIN_STEPS}, RES={RESOLUTION}, RANK={RANK}")
    print("Starting training...\n")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Print last lines (useful summary)
        lines = result.stdout.split("\n")
        print("\n".join(lines[-25:]))

        weight_file = lora_class_dir / "pytorch_lora_weights.safetensors"
        if not weight_file.exists():
            print(f"\n⚠️  Weight file not found: {weight_file}")
            return False

        # Copy to root LoRA_W with mapped filename (overwrite old)
        target_file = LORA_OUTPUT_DIR / CLASS_TO_LORA_FILE[class_name]
        shutil.copy2(weight_file, target_file)
        print(f"\n✓ Weight copied to: {target_file}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed for {class_name}")
        print(f"Return code: {e.returncode}")
        # If stdout captured, show tail to debug
        if hasattr(e, "stdout") and e.stdout:
            tail = e.stdout.split("\n")[-40:]
            print("\n".join(tail))
        return False


# ---------------------------------------------------------
# Step 3: Generate images for a class (overwrite old images)
# ---------------------------------------------------------
def generate_images_for_class(class_name: str) -> bool:
    print(f"\n{'=' * 80}")
    print(f"Generating images: {class_name}")
    print(f"{'=' * 80}")

    class_gen_dir = GENERATED_DIR / class_name
    if class_gen_dir.exists():
        print(f"Deleting old images: {class_gen_dir}")
        shutil.rmtree(class_gen_dir)
    class_gen_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
    from tqdm import tqdm

    print("Loading Stable Diffusion model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    lora_file = LORA_OUTPUT_DIR / CLASS_TO_LORA_FILE[class_name]
    if not lora_file.exists():
        print(f"⚠️  LoRA weight not found: {lora_file}")
        return False

    print(f"Loading LoRA: {lora_file.name}")
    # weight_name expects file inside the directory passed to load_lora_weights
    pipe.load_lora_weights(str(LORA_OUTPUT_DIR), weight_name=CLASS_TO_LORA_FILE[class_name])

    prompt = build_generation_prompt(class_name)

    print(f"Generating {GEN_NUM_IMAGES} images...")
    print(f"Prompt: {prompt[:120]}...")
    print(f"steps={GEN_STEPS}, guidance={GEN_GUIDANCE}, batch={GEN_BATCH_SIZE}")

    for i in tqdm(range(0, GEN_NUM_IMAGES, GEN_BATCH_SIZE), desc=class_name):
        batch = min(GEN_BATCH_SIZE, GEN_NUM_IMAGES - i)

        generators = [
            torch.Generator(device="cuda").manual_seed(SEED + i + j)
            for j in range(batch)
        ]

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            images = pipe(
                [prompt] * batch,
                negative_prompt=[NEGATIVE_PROMPT] * batch,
                num_inference_steps=GEN_STEPS,
                guidance_scale=GEN_GUIDANCE,
                generator=generators,
                height=512,
                width=512,
            ).images

        for j, img in enumerate(images):
            seed = SEED + i + j
            img_name = f"{class_name}_{i+j:05d}_seed{seed}.jpg"
            img.save(class_gen_dir / img_name, quality=95, optimize=True)

    print(f"✓ Generated {GEN_NUM_IMAGES} images: {class_gen_dir}")

    # cleanup
    del pipe
    torch.cuda.empty_cache()
    return True


def main():
    print("\n" + "=" * 80)
    print("RETRAIN LORA FOR ALL CLASSES (15) + REGENERATE IMAGES")
    print("=" * 80)

    print("\nClasses:")
    for i, cls in enumerate(ALL_CLASSES, 1):
        print(f"  {i:02d}. {cls}")

    print("\nTraining config:")
    print(f"  BASE_MODEL: {BASE_MODEL}")
    print(f"  RESOLUTION: {RESOLUTION}")
    print(f"  N_SHOT: {N_SHOT}")
    print(f"  MAX_TRAIN_STEPS: {MAX_TRAIN_STEPS}")
    print(f"  LR: {LR}, RANK: {RANK}")
    print("\nGeneration config:")
    print(f"  NUM_IMAGES/CLASS: {GEN_NUM_IMAGES}")
    print(f"  STEPS: {GEN_STEPS}, GUIDANCE: {GEN_GUIDANCE}, BATCH: {GEN_BATCH_SIZE}")
    print("\nThis run will overwrite old weights & images.\n")

    # Step 1
    prepare_fewshot_data()

    # Step 2
    print("\n" + "=" * 80)
    print("STEP 2: Training LoRA (ALL classes)")
    print("=" * 80)

    LORA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success_classes = []
    failed_train = []

    for class_name in ALL_CLASSES:
        ok = train_lora_for_class(class_name, gpu_id=0)
        if ok:
            success_classes.append(class_name)
        else:
            failed_train.append(class_name)

    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"✓ Success: {len(success_classes)}/{len(ALL_CLASSES)}")
    if failed_train:
        print(f"❌ Failed ({len(failed_train)}): {failed_train}")

    # Step 3
    if success_classes:
        print("\n" + "=" * 80)
        print("STEP 3: Generating Images (for successfully trained classes)")
        print("=" * 80)

        GENERATED_DIR.mkdir(parents=True, exist_ok=True)

        for class_name in success_classes:
            try:
                generate_images_for_class(class_name)
            except Exception as e:
                print(f"❌ Failed to generate for {class_name}: {e}")

    # Save metadata
    metadata = {
        "base_model": BASE_MODEL,
        "resolution": RESOLUTION,
        "n_shot": N_SHOT,
        "max_train_steps": MAX_TRAIN_STEPS,
        "lr": LR,
        "rank": RANK,
        "train_batch_size": TRAIN_BATCH,
        "grad_accum": GRAD_ACCUM,
        "seed": SEED,
        "gen_num_images_per_class": GEN_NUM_IMAGES,
        "gen_steps": GEN_STEPS,
        "gen_guidance": GEN_GUIDANCE,
        "gen_batch_size": GEN_BATCH_SIZE,
        "classes": ALL_CLASSES,
        "lora_dir": str(LORA_OUTPUT_DIR),
        "generated_dir": str(GENERATED_DIR),
        "method": "DreamBooth LoRA training per class + text-to-image generation (no ControlNet)",
    }
    with open(GENERATED_DIR / "retrain_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("✓ RETRAIN COMPLETE!")
    print("=" * 80)
    print(f"LoRA weights dir: {LORA_OUTPUT_DIR}")
    print(f"Generated images dir: {GENERATED_DIR}")
    print(f"Metadata: {GENERATED_DIR / 'retrain_metadata.json'}")
    print("\nNext steps:")
    print("  1) Verify generated images quality per class")
    print("  2) Rebuild datasets (real-only / synthetic-only / balanced)")
    print("  3) Retrain classifiers and compare results\n")


if __name__ == "__main__":
    main()
