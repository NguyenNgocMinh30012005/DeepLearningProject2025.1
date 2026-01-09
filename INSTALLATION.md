# Installation Guide

## System Requirements

- Python 3.8+
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM
- 50GB+ free disk space

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/NguyenNgocMinh30012005/DeepLearningProject2025.1.git
cd DeepLearningProject2025.1
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download Dataset

Download PlantVillage dataset and place in `dataset_original/PlantVillage/`

### 5. Verify Installation

```bash
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python3 -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
```

## Quick Start

### Generate Synthetic Images

```bash
cd scripts
python3 generate_with_lora.py --n_per_class 1024 --batch_size 4
```

### Train Classifier

```bash
cd scripts
bash run_experiments.sh
```

### Visualize Results

```bash
cd scripts
python3 visualize_all_experiments.py
```

## GPU Setup

For NVIDIA GPUs:
```bash
# Check GPU
nvidia-smi

# Verify PyTorch can see GPU
python3 -c "import torch; print(torch.cuda.device_count())"
```

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in scripts
- Use gradient checkpointing
- Use mixed precision training

### Module Not Found
```bash
pip install -r requirements.txt --upgrade
```

### Permission Denied
```bash
chmod +x scripts/*.sh
```

## Project Structure

```
.
├── scripts/              # All Python & Shell scripts
├── reports/              # Experiment reports
├── final_visualizations/ # Results plots
├── logs/                 # Training logs
├── dataset_original/     # Original dataset (not in repo)
├── generated_images/     # Synthetic images (not in repo)
├── LoRA_W/              # LoRA weights (not in repo)
└── experiments_results/  # Experiment outputs (not in repo)
```

## Next Steps

1. Read `README.md` for project overview
2. Check `reports/EXPERIMENTS_FINAL_REPORT.md` for results
3. Explore `scripts/` for code
4. Run experiments with `scripts/run_experiments.sh`

## Support

For issues or questions, please open an issue on GitHub.
