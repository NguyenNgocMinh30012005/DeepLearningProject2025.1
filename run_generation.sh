#!/bin/bash

# ========================================
# IMAGE GENERATION WITH DATADREAM LORA
# ========================================
# Generate 1,024 images per class using trained LoRA weights
# With ENHANCED BOTANICAL PROMPTS for realistic leaf images

set -e

# Configuration
GPU=0
DATASET="plantvillage"
N_SHOT=16
N_TEMPLATE=1
FEWSHOT_SEED="seed0"
SD_VERSION="sd1.5"
MODE="datadream"

# Generation settings
N_IMG_PER_CLASS=1024  # Generate 1,024 images per class
GUIDANCE_SCALE=2.0    # Lower = more diverse, Higher = more prompt-adherent
NUM_INFERENCE_STEPS=50
BS=4                  # Batch size (reduced from 10 to 4 for memory)
SEED=42

# LoRA settings (must match training)
DATADREAM_LR=0.0001
DATADREAM_EPOCH=200
DATADREAM_TRAIN_TEXT_ENCODER="True"

# Paths
DATADREAM_DIR="/workspace/DataDream/outputs/${DATASET}"
SYNTH_DATA_DIR="/workspace/DataDream/data/${DATASET}/synthetic"
GENERATE_DIR="/workspace/DataDream/generate"

# Class names (15 classes)
CLASSES=(
    "Pepper__bell___Bacterial_spot"
    "Pepper__bell___healthy"
    "Potato___Early_blight"
    "Potato___Late_blight"
    "Potato___healthy"
    "Tomato_Bacterial_spot"
    "Tomato_Early_blight"
    "Tomato_Late_blight"
    "Tomato_Leaf_Mold"
    "Tomato_Septoria_leaf_spot"
    "Tomato_Spider_mites_Two_spotted_spider_mite"
    "Tomato__Target_Spot"
    "Tomato__Tomato_YellowLeaf__Curl_Virus"
    "Tomato__Tomato_mosaic_virus"
    "Tomato_healthy"
)

NUM_CLASSES=${#CLASSES[@]}

# Create output directory
mkdir -p "$SYNTH_DATA_DIR"

# Log file
LOG_FILE="/workspace/logs/generation.log"
mkdir -p /workspace/logs
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "🎨 DATADREAM IMAGE GENERATION"
echo "=========================================="
echo ""
echo "📋 Configuration:"
echo "  - Dataset: $DATASET"
echo "  - Classes: $NUM_CLASSES"
echo "  - Images per class: $N_IMG_PER_CLASS"
echo "  - Total images: $((NUM_CLASSES * N_IMG_PER_CLASS))"
echo "  - SD Model: $SD_VERSION"
echo "  - Guidance Scale: $GUIDANCE_SCALE"
echo "  - Inference Steps: $NUM_INFERENCE_STEPS"
echo "  - Batch Size: $BS"
echo "  - GPU: $GPU"
echo ""
echo "🎨 ENHANCED PROMPTS:"
echo "  - Professional botanical photography style"
echo "  - High-resolution macro details"
echo "  - Natural lighting conditions"
echo "  - Clear disease symptom visibility"
echo "  - Agricultural documentation quality"
echo ""
echo "Starting generation at: $(date)"
echo "=========================================="
echo ""

# Change to generate directory
cd "$GENERATE_DIR"

# Track progress
TOTAL_IMAGES=$((NUM_CLASSES * N_IMG_PER_CLASS))
CURRENT_IMAGES=0

# Generate for each class
for CLASS_IDX in $(seq 0 $((NUM_CLASSES - 1))); do
    CLASS_NAME="${CLASSES[$CLASS_IDX]}"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🎨 Generating Class $((CLASS_IDX + 1))/$NUM_CLASSES: $CLASS_NAME"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    # Check if LoRA weights exist
    LORA_PATH="$DATADREAM_DIR/$DATASET/shot${N_SHOT}_${FEWSHOT_SEED}_tpl${N_TEMPLATE}/lr${DATADREAM_LR}_epoch${DATADREAM_EPOCH}/${CLASS_NAME}/pytorch_lora_weights.safetensors"
    
    if [ ! -f "$LORA_PATH" ]; then
        echo "❌ ERROR: LoRA weights not found at: $LORA_PATH"
        echo "   Skipping this class..."
        continue
    fi
    
    echo "✓ LoRA weights found"
    echo "📊 Progress: $CURRENT_IMAGES/$TOTAL_IMAGES images generated ($((CURRENT_IMAGES * 100 / TOTAL_IMAGES))%)"
    echo ""
    
    START_TIME=$(date +%s)
    
    # Run generation
    CUDA_VISIBLE_DEVICES=$GPU python3 generate.py \
        --seed=$SEED \
        --sd_version=$SD_VERSION \
        --mode=$MODE \
        --guidance_scale=$GUIDANCE_SCALE \
        --num_inference_steps=$NUM_INFERENCE_STEPS \
        --n_img_per_class=$N_IMG_PER_CLASS \
        --bs=$BS \
        --n_shot=$N_SHOT \
        --n_template=$N_TEMPLATE \
        --dataset=$DATASET \
        --fewshot_seed=$FEWSHOT_SEED \
        --datadream_lr=$DATADREAM_LR \
        --datadream_epoch=$DATADREAM_EPOCH \
        --datadream_train_text_encoder=$DATADREAM_TRAIN_TEXT_ENCODER \
        --target_class_idx=$CLASS_IDX \
        --is_tqdm=True
    
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    
    CURRENT_IMAGES=$((CURRENT_IMAGES + N_IMG_PER_CLASS))
    
    echo ""
    echo "✅ Class $CLASS_NAME completed in ${ELAPSED}s"
    echo "📊 Total progress: $CURRENT_IMAGES/$TOTAL_IMAGES images ($((CURRENT_IMAGES * 100 / TOTAL_IMAGES))%)"
    
    # Estimate remaining time
    if [ $CLASS_IDX -gt 0 ]; then
        AVG_TIME_PER_CLASS=$((ELAPSED))
        REMAINING_CLASSES=$((NUM_CLASSES - CLASS_IDX - 1))
        EST_REMAINING_TIME=$((AVG_TIME_PER_CLASS * REMAINING_CLASSES))
        EST_REMAINING_MIN=$((EST_REMAINING_TIME / 60))
        echo "⏱️  Estimated time remaining: ${EST_REMAINING_MIN} minutes"
    fi
done

echo ""
echo "=========================================="
echo "✅ GENERATION COMPLETE!"
echo "=========================================="
echo ""
echo "📊 Summary:"
echo "  - Total images generated: $TOTAL_IMAGES"
echo "  - Output directory: $SYNTH_DATA_DIR"
echo "  - Classes: $NUM_CLASSES"
echo ""
echo "Completed at: $(date)"
echo ""

# Verify generated images
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Verifying generated images..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for CLASS_NAME in "${CLASSES[@]}"; do
    CLASS_DIR="$SYNTH_DATA_DIR/$CLASS_NAME"
    if [ -d "$CLASS_DIR" ]; then
        COUNT=$(find "$CLASS_DIR" -type f \( -name "*.jpg" -o -name "*.png" \) | wc -l)
        printf "  %-50s %5d images\n" "$CLASS_NAME:" "$COUNT"
    else
        printf "  %-50s %5s\n" "$CLASS_NAME:" "❌ Not found"
    fi
done

echo ""
echo "🎉 All done! Ready for classifier training!"
