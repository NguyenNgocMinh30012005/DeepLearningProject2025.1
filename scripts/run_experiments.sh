#!/bin/bash
#
# Script chạy experiments WITHOUT PRETRAINED WEIGHTS (from scratch)
# So sánh với version có pretrained
#

set -e

# ============================================
# Configuration
# ============================================
REAL_DIR="/workspace/dataset_original/PlantVillage"
SYNTH_DIR="/workspace/generated_images"
DATA_BASE="/workspace/experiments_data"
RESULTS_BASE="/workspace/experiments_results_no_pretrain"

EPOCHS=50
BATCH_SIZE=32
LR=0.001
NUM_WORKERS=4

echo "========================================"
echo "EXPERIMENTS WITHOUT PRETRAINED WEIGHTS"
echo "Training from SCRATCH"
echo "========================================"

# ============================================
# Exp1: Train on Synthetic, Test on Real
# ============================================
echo ""
echo "========================================"
echo "EXPERIMENT 1: Train Syn, Val Syn, Test Real (NO PRETRAIN)"
echo "========================================"

# Exp1.1: ResNet18
echo ""
echo "Exp1.1: ResNet18 (Syn -> Real) - FROM SCRATCH"
python3 /workspace/train_classifier.py \
    --train_dir "$SYNTH_DIR" \
    --val_dir "$SYNTH_DIR" \
    --test_dir "$DATA_BASE/real_split/test" \
    --output_dir "$RESULTS_BASE" \
    --exp_name "exp1_resnet18_syn_no_pretrain" \
    --model resnet18 \
    --num_classes 15 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --optimizer adam \
    --scheduler cosine \
    --num_workers $NUM_WORKERS

# Exp1.2: ResNet50
echo ""
echo "Exp1.2: ResNet50 (Syn -> Real) - FROM SCRATCH"
python3 /workspace/train_classifier.py \
    --train_dir "$SYNTH_DIR" \
    --val_dir "$SYNTH_DIR" \
    --test_dir "$DATA_BASE/real_split/test" \
    --output_dir "$RESULTS_BASE" \
    --exp_name "exp1_resnet50_syn_no_pretrain" \
    --model resnet50 \
    --num_classes 15 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --optimizer adam \
    --scheduler cosine \
    --num_workers $NUM_WORKERS

# ============================================
# Exp2: Train on Real+Syn (Balanced), Test on Real
# ============================================
echo ""
echo "========================================"
echo "EXPERIMENT 2: Train Real+Syn (Balanced), Test Real (NO PRETRAIN)"
echo "========================================"

# Exp2.1: ResNet18
echo ""
echo "Exp2.1: ResNet18 (Balanced Real+Syn -> Real) - FROM SCRATCH"
python3 /workspace/train_classifier.py \
    --train_dir "$DATA_BASE/balanced_train" \
    --val_dir "$DATA_BASE/balanced_val" \
    --test_dir "$DATA_BASE/real_split/test" \
    --output_dir "$RESULTS_BASE" \
    --exp_name "exp2_resnet18_balanced_no_pretrain" \
    --model resnet18 \
    --num_classes 15 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --optimizer adam \
    --scheduler cosine \
    --num_workers $NUM_WORKERS

# ============================================
# Baseline - Train only on Real
# ============================================
echo ""
echo "========================================"
echo "BASELINE: Train Real Only (NO PRETRAIN)"
echo "========================================"

# Baseline: ResNet18
echo ""
echo "Baseline: ResNet18 (Real Only) - FROM SCRATCH"
python3 /workspace/train_classifier.py \
    --train_dir "$DATA_BASE/real_split/train" \
    --val_dir "$DATA_BASE/real_split/val" \
    --test_dir "$DATA_BASE/real_split/test" \
    --output_dir "$RESULTS_BASE" \
    --exp_name "baseline_resnet18_real_no_pretrain" \
    --model resnet18 \
    --num_classes 15 \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --optimizer adam \
    --scheduler cosine \
    --num_workers $NUM_WORKERS


