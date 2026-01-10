#!/bin/bash
#
# DataDream Full Pipeline for PlantVillage
# Usage: bash run_datadream_pipeline.sh
#

set -e  # Exit on error

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 DATADREAM FULL PIPELINE FOR PLANTVILLAGE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Configuration
GPU=0
DATASET="plantvillage"
N_SHOT=16
FEWSHOT_SEED="seed0"
N_CLASSES=15
N_IMG_PER_CLASS=1024  # Same as before
GUIDANCE_SCALE=2.0
LR=1e-4
EPOCHS=200  # Same as LoRA training before (1200 steps ≈ 200 epochs with small data)

DATADREAM_DIR="/workspace/DataDream-repo"
DATA_DIR="${DATADREAM_DIR}/data/${DATASET}"

echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  N-shot: $N_SHOT"
echo "  Classes: $N_CLASSES"
echo "  Images per class: $N_IMG_PER_CLASS"
echo "  LoRA LR: $LR"
echo "  LoRA Epochs: $EPOCHS"
echo ""

# Check data
if [ ! -d "$DATA_DIR/real_train_fewshot/$FEWSHOT_SEED" ]; then
    echo "❌ Error: Few-shot data not found at $DATA_DIR/real_train_fewshot/$FEWSHOT_SEED"
    echo "Please run: python3 scripts/prepare_datadream_data.py"
    exit 1
fi

echo "✓ Few-shot data found"

# Update local.yaml
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 Step 0: Update Configuration Files"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Update DataDream/local.yaml
cat > ${DATADREAM_DIR}/DataDream/local.yaml << YAML_END
fewshot_data_dir: 
    ${DATASET}: ${DATA_DIR}/real_train_fewshot

model_dir: 
    sd1.5: /workspace/models/stable-diffusion-1-5
YAML_END

echo "✓ Updated DataDream/local.yaml"

# Update generate/local.yaml
cat > ${DATADREAM_DIR}/generate/local.yaml << YAML_END
fewshot_data_dir: 
    ${DATASET}: ${DATA_DIR}/real_train_fewshot

model_dir: 
    sd1.5: /workspace/models/stable-diffusion-1-5

datadream_dir:
    ${DATASET}: ${DATADREAM_DIR}/outputs/${DATASET}

synth_data_dir:
    ${DATASET}: ${DATA_DIR}/synthetic
YAML_END

echo "✓ Updated generate/local.yaml"

# Update classify/local.yaml
cat > ${DATADREAM_DIR}/classify/local.yaml << YAML_END
real_train_data_dir:
    ${DATASET}: ${DATA_DIR}/real_test

real_test_data_dir:
    ${DATASET}: ${DATA_DIR}/real_test

real_train_fewshot_data_dir:
    ${DATASET}: ${DATA_DIR}/real_train_fewshot

synth_data_dir:
    ${DATASET}: ${DATA_DIR}/synthetic
YAML_END

echo "✓ Updated classify/local.yaml"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Step 1: Train LoRA weights for ALL classes"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

mkdir -p ${DATADREAM_DIR}/outputs/${DATASET}

cd ${DATADREAM_DIR}/DataDream

# Train for each class
for CLASS_IDX in $(seq 0 $((N_CLASSES-1))); do
    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "Training LoRA for class ${CLASS_IDX}/${N_CLASSES}"
    echo "────────────────────────────────────────────────────────────"
    
    CUDA_VISIBLE_DEVICES=$GPU accelerate launch datadream.py \
        --dataset=$DATASET \
        --target_class_idx=$CLASS_IDX \
        --fewshot_seed=$FEWSHOT_SEED \
        --n_shot=$N_SHOT \
        --n_template=1 \
        --train_text_encoder=True \
        --resume_from_checkpoint=None \
        --train_batch_size=4 \
        --gradient_accumulation_steps=2 \
        --learning_rate=$LR \
        --lr_scheduler="cosine" \
        --lr_warmup_steps=100 \
        --num_train_epochs=$EPOCHS \
        --report_to="tensorboard" \
        --is_tqdm=True \
        --output_dir=../outputs/${DATASET} \
        --mixed_precision="no"
    
    echo "✓ LoRA training completed for class $CLASS_IDX"
done

echo ""
echo "✅ All LoRA weights trained!"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎨 Step 2: Generate Synthetic Images"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd ${DATADREAM_DIR}/generate

mkdir -p ${DATA_DIR}/synthetic

CUDA_VISIBLE_DEVICES=$GPU python generate.py \
    --sd_version=sd1.5 \
    --mode=datadream \
    --is_dataset_wise_model=False \
    --dataset=$DATASET \
    --n_shot=$N_SHOT \
    --fewshot_seed=$FEWSHOT_SEED \
    --bs=10 \
    --n_img_per_class=$N_IMG_PER_CLASS \
    --guidance_scale=$GUIDANCE_SCALE \
    --n_template=1 \
    --datadream_lr=$LR \
    --datadream_epoch=$EPOCHS \
    --n_set_split=1 \
    --split_idx=0

echo ""
echo "✅ Synthetic images generated!"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎓 Step 3: Train Classifier (Pure Synthetic)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd ${DATADREAM_DIR}/classify

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --model_type=clip \
    --dataset=$DATASET \
    --is_synth_train=True \
    --is_pooled_fewshot=False \
    --is_dataset_wise=False \
    --datadream_lr=$LR \
    --datadream_epoch=$EPOCHS \
    --n_shot=$N_SHOT \
    --fewshot_seed=$FEWSHOT_SEED \
    --n_img_per_cls=$N_IMG_PER_CLASS \
    --sd_version="sd1.5" \
    --n_template=1 \
    --guidance_scale=$GUIDANCE_SCALE \
    --lambda_1=0.8 \
    --epochs=50 \
    --warmup_epochs=5 \
    --lr=1e-3 \
    --wd=1e-4 \
    --min_lr=1e-6 \
    --is_mix_aug=True \
    --log=tensorboard

echo ""
echo "✅ Classifier trained (Pure Synthetic)!"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎓 Step 4: Train Classifier (Synthetic + Few-shot Real)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CUDA_VISIBLE_DEVICES=$GPU python main.py \
    --model_type=clip \
    --dataset=$DATASET \
    --is_synth_train=True \
    --is_pooled_fewshot=True \
    --is_dataset_wise=False \
    --datadream_lr=$LR \
    --datadream_epoch=$EPOCHS \
    --n_shot=$N_SHOT \
    --fewshot_seed=$FEWSHOT_SEED \
    --n_img_per_cls=$N_IMG_PER_CLASS \
    --sd_version="sd1.5" \
    --n_template=1 \
    --guidance_scale=$GUIDANCE_SCALE \
    --lambda_1=0.8 \
    --epochs=50 \
    --warmup_epochs=5 \
    --lr=1e-3 \
    --wd=1e-4 \
    --min_lr=1e-6 \
    --is_mix_aug=True \
    --log=tensorboard

echo ""
echo "✅ Classifier trained (Synthetic + Real)!"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 DATADREAM PIPELINE COMPLETED!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "Results:"
echo "  LoRA weights: ${DATADREAM_DIR}/outputs/${DATASET}/"
echo "  Synthetic images: ${DATA_DIR}/synthetic/"
echo "  Classifier logs: ${DATADREAM_DIR}/classify/logs/"
echo ""
