#!/usr/bin/env python3
"""
Comprehensive Visualization cho TẤT CẢ EXPERIMENTS
- Training curves (loss, accuracy)
- Test metrics comparison
- Per-class metrics (Precision, Recall, F1)
- Confusion matrices
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Experiment paths
EXPERIMENTS = {
    'Exp1.1: ResNet18\n(Syn→Real)': '/workspace/results_retrained/exp1_syn2real_resnet18/exp1_syn2real_resnet18_retrained',
    'Exp1.2: ResNet50\n(Syn→Real)': '/workspace/results_retrained/exp1_syn2real_resnet50/exp1_syn2real_resnet50_retrained',
    'Exp2: ResNet18\n(Balanced→Real)': '/workspace/results_retrained/exp2_balanced_resnet18/exp2_balanced_resnet18_retrained',
    'Baseline: ResNet18\n(Real→Real)': '/workspace/results_retrained/baseline_real_only/baseline_real_only_no_pretrain',
}

OUTPUT_DIR = Path('/workspace/final_visualizations')
OUTPUT_DIR.mkdir(exist_ok=True)

def load_experiment_data(exp_path):
    """Load training history and test results"""
    exp_path = Path(exp_path)
    
    history = None
    history_path = exp_path / 'training_history.json'
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
    
    with open(exp_path / 'test_results.json') as f:
        test_results = json.load(f)
    
    return history, test_results

def plot_training_curves():
    """Plot training curves for all experiments"""
    print("\n📊 Creating Training Curves...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Training Curves - All Experiments', fontsize=16, fontweight='bold', y=0.995)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    # Plot Loss
    ax = axes[0, 0]
    for (exp_name, exp_path), color in zip(EXPERIMENTS.items(), colors):
        history, _ = load_experiment_data(exp_path)
        if history is None:
            continue  # Skip if no training history
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], label=f'{exp_name} (Train)', 
                color=color, linewidth=2, alpha=0.8)
        ax.plot(epochs, history['val_loss'], label=f'{exp_name} (Val)', 
                color=color, linewidth=2, linestyle='--', alpha=0.6)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=11, fontweight='bold')
    ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot Accuracy
    ax = axes[0, 1]
    for (exp_name, exp_path), color in zip(EXPERIMENTS.items(), colors):
        history, _ = load_experiment_data(exp_path)
        if history is None:
            continue  # Skip if no training history
        epochs = range(1, len(history['train_acc']) + 1)
        ax.plot(epochs, history['train_acc'], label=f'{exp_name} (Train)', 
                color=color, linewidth=2, alpha=0.8)
        ax.plot(epochs, history['val_acc'], label=f'{exp_name} (Val)', 
                color=color, linewidth=2, linestyle='--', alpha=0.6)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([50, 100])
    
    # Plot Train Loss (zoomed, last 20 epochs)
    ax = axes[1, 0]
    for (exp_name, exp_path), color in zip(EXPERIMENTS.items(), colors):
        history, _ = load_experiment_data(exp_path)
        if history is None:
            continue  # Skip if no training history
        epochs = range(31, 51)
        ax.plot(epochs, history['train_loss'][30:50], label=exp_name, 
                color=color, linewidth=2.5, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Train Loss', fontsize=11, fontweight='bold')
    ax.set_title('Training Loss (Epochs 31-50, Zoomed)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot Val Accuracy (zoomed, last 20 epochs)
    ax = axes[1, 1]
    for (exp_name, exp_path), color in zip(EXPERIMENTS.items(), colors):
        history, _ = load_experiment_data(exp_path)
        if history is None:
            continue  # Skip if no training history
        epochs = range(31, 51)
        ax.plot(epochs, history['val_acc'][30:50], label=exp_name, 
                color=color, linewidth=2.5, marker='o', markersize=3)
    
    ax.set_xlabel('Epoch', fontsize=11, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Validation Accuracy (Epochs 31-50, Zoomed)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([95, 100])
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'training_curves_all.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: training_curves_all.png")
    plt.close()

def plot_test_metrics_comparison():
    """Plot test metrics comparison"""
    print("\n📊 Creating Test Metrics Comparison...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Test Performance - All Experiments', fontsize=16, fontweight='bold')
    
    # Collect data
    exp_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for exp_name, exp_path in EXPERIMENTS.items():
        _, test_results = load_experiment_data(exp_path)
        exp_names.append(exp_name.replace('\n', ' '))
        accuracies.append(test_results['test_accuracy'])
        precisions.append(test_results['precision_weighted'] * 100)
        recalls.append(test_results['recall_weighted'] * 100)
        f1_scores.append(test_results['f1_weighted'] * 100)
    
    # Plot 1: Overall Metrics
    ax = axes[0]
    x = np.arange(len(exp_names))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#4ECDC4', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#45B7D1', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#FFA07A', alpha=0.8)
    
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Overall Test Metrics Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.split('(')[0].strip() for name in exp_names], fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([98, 100])
    
    # Add value labels on bars
    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    # Plot 2: Ranking
    ax = axes[1]
    sorted_indices = np.argsort(accuracies)[::-1]
    sorted_names = [exp_names[i].split('(')[0].strip() for i in sorted_indices]
    sorted_accs = [accuracies[i] for i in sorted_indices]
    
    colors_rank = ['#FFD700', '#C0C0C0', '#CD7F32', '#87CEEB']
    bars = ax.barh(range(len(sorted_names)), sorted_accs, color=colors_rank, alpha=0.8)
    
    ax.set_xlabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Test Accuracy Ranking', fontsize=13, fontweight='bold')
    ax.set_yticks(range(len(sorted_names)))
    ax.set_yticklabels(sorted_names, fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim([98.5, 100])
    
    # Add value labels
    for i, (bar, acc) in enumerate(zip(bars, sorted_accs)):
        rank_emoji = ['🥇', '🥈', '🥉', ''][i]
        ax.text(acc, bar.get_y() + bar.get_height()/2., 
               f'  {acc:.2f}% {rank_emoji}', 
               ha='left', va='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'test_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: test_metrics_comparison.png")
    plt.close()

def plot_per_class_metrics():
    """Plot per-class precision, recall, F1"""
    print("\n📊 Creating Per-Class Metrics...")
    
    # Get class names
    _, test_results = load_experiment_data(list(EXPERIMENTS.values())[0])
    class_names = list(test_results['per_class_metrics'].keys())
    
    # Shorten class names for display
    short_names = [name.replace('Tomato_', 'T_').replace('Potato_', 'P_')
                   .replace('Pepper__bell___', 'Pep_').replace('__', '_')
                   for name in class_names]
    
    # Create subplots
    fig, axes = plt.subplots(3, 1, figsize=(18, 16))
    fig.suptitle('Per-Class Metrics - All Experiments', fontsize=16, fontweight='bold', y=0.995)
    
    metrics_to_plot = ['precision', 'recall', 'f1']
    metric_titles = ['Precision', 'Recall', 'F1-Score']
    
    x = np.arange(len(class_names))
    width = 0.2
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    
    for metric_idx, (metric, title) in enumerate(zip(metrics_to_plot, metric_titles)):
        ax = axes[metric_idx]
        
        for exp_idx, (exp_name, exp_path) in enumerate(EXPERIMENTS.items()):
            _, test_results = load_experiment_data(exp_path)
            
            values = [test_results['per_class_metrics'][cls][metric] * 100 
                     for cls in class_names]
            
            offset = (exp_idx - 1.5) * width
            ax.bar(x + offset, values, width, 
                  label=exp_name.replace('\n', ' '), 
                  color=colors[exp_idx], alpha=0.8)
        
        ax.set_ylabel(f'{title} (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'Per-Class {title}', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=9, loc='lower left')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([95, 101])
        ax.axhline(y=99, color='red', linestyle='--', alpha=0.3, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'per_class_metrics.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: per_class_metrics.png")
    plt.close()

def plot_confusion_matrices():
    """Plot confusion matrices for all experiments"""
    print("\n📊 Creating Confusion Matrices Grid...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 18))
    fig.suptitle('Confusion Matrices - All Experiments', fontsize=16, fontweight='bold', y=0.995)
    
    # Get class names
    _, test_results = load_experiment_data(list(EXPERIMENTS.values())[0])
    class_names = list(test_results['per_class_metrics'].keys())
    
    # Shorten class names
    short_names = [name.replace('Tomato_', 'T.').replace('Potato_', 'Po.')
                   .replace('Pepper__bell___', 'Pe.').replace('__', '_')[:15]
                   for name in class_names]
    
    for idx, (exp_name, exp_path) in enumerate(EXPERIMENTS.items()):
        ax = axes[idx // 2, idx % 2]
        
        # Load confusion matrix
        cm_path = Path(exp_path) / 'confusion_matrix.png'
        if cm_path.exists():
            # Read existing confusion matrix
            import matplotlib.image as mpimg
            img = mpimg.imread(str(cm_path))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(exp_name.replace('\n', ' '), fontsize=13, fontweight='bold', pad=10)
        else:
            ax.text(0.5, 0.5, 'Confusion Matrix\nNot Available', 
                   ha='center', va='center', fontsize=14)
            ax.set_title(exp_name.replace('\n', ' '), fontsize=13, fontweight='bold')
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'confusion_matrices_grid.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: confusion_matrices_grid.png")
    plt.close()

def create_summary_table():
    """Create summary table image"""
    print("\n📊 Creating Summary Table...")
    
    # Collect data
    data = []
    for exp_name, exp_path in EXPERIMENTS.items():
        _, test_results = load_experiment_data(exp_path)
        
        data.append({
            'Experiment': exp_name.replace('\n', ' '),
            'Accuracy': f"{test_results['test_accuracy']:.2f}%",
            'Precision': f"{test_results['precision_weighted']*100:.2f}%",
            'Recall': f"{test_results['recall_weighted']*100:.2f}%",
            'F1-Score': f"{test_results['f1_weighted']*100:.2f}%",
            'Test Loss': f"{test_results['test_loss']:.4f}"
        })
    
    df = pd.DataFrame(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values, colLabels=df.columns,
                    cellLoc='center', loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(df.columns)):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style rows with alternating colors
    colors = ['#f0f0f0', '#ffffff']
    for i in range(1, len(df) + 1):
        for j in range(len(df.columns)):
            table[(i, j)].set_facecolor(colors[i % 2])
    
    # Highlight best values
    best_acc_idx = df['Accuracy'].str.rstrip('%').astype(float).idxmax() + 1
    table[(best_acc_idx, 1)].set_facecolor('#FFD700')
    table[(best_acc_idx, 1)].set_text_props(weight='bold')
    
    plt.title('Summary Table - All Experiments', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(OUTPUT_DIR / 'summary_table.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: summary_table.png")
    plt.close()

def main():
    print("="*80)
    print("COMPREHENSIVE VISUALIZATION - ALL EXPERIMENTS")
    print("="*80)
    
    print("\n📁 Experiments:")
    for exp_name, exp_path in EXPERIMENTS.items():
        print(f"   • {exp_name.replace(chr(10), ' ')}")
    
    print(f"\n📂 Output directory: {OUTPUT_DIR}")
    
    # Generate all visualizations
    plot_training_curves()
    plot_test_metrics_comparison()
    plot_per_class_metrics()
    plot_confusion_matrices()
    create_summary_table()
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS COMPLETED!")
    print("="*80)
    print(f"\n📊 Generated files in: {OUTPUT_DIR}/")
    print("   1. training_curves_all.png")
    print("   2. test_metrics_comparison.png")
    print("   3. per_class_metrics.png")
    print("   4. confusion_matrices_grid.png")
    print("   5. summary_table.png")
    print("\n🎉 Done!")

if __name__ == '__main__':
    main()
