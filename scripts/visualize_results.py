#!/usr/bin/env python3
"""
Visualization script - Vẽ tất cả đồ thị cho experiments
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

def load_history(exp_dir):
    """Load training history"""
    history_path = os.path.join(exp_dir, 'training_history.json')
    if not os.path.exists(history_path):
        return None
    with open(history_path, 'r') as f:
        return json.load(f)

def load_test_results(exp_dir):
    """Load test results"""
    results_path = os.path.join(exp_dir, 'test_results.json')
    if not os.path.exists(results_path):
        return None
    with open(results_path, 'r') as f:
        return json.load(f)

def plot_training_curves(experiments, output_dir):
    """
    Vẽ đồ thị loss và accuracy cho tất cả experiments
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Train Loss
    ax = axes[0, 0]
    for exp_name, history in experiments.items():
        if history:
            epochs = range(1, len(history['train_loss']) + 1)
            ax.plot(epochs, history['train_loss'], marker='o', 
                   label=exp_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Train Loss', fontsize=12, fontweight='bold')
    ax.set_title('Training Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Val Loss
    ax = axes[0, 1]
    for exp_name, history in experiments.items():
        if history:
            epochs = range(1, len(history['val_loss']) + 1)
            ax.plot(epochs, history['val_loss'], marker='s', 
                   label=exp_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Loss Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Train Accuracy
    ax = axes[1, 0]
    for exp_name, history in experiments.items():
        if history:
            epochs = range(1, len(history['train_acc']) + 1)
            ax.plot(epochs, history['train_acc'], marker='o', 
                   label=exp_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Train Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Training Accuracy Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Val Accuracy
    ax = axes[1, 1]
    for exp_name, history in experiments.items():
        if history:
            epochs = range(1, len(history['val_acc']) + 1)
            ax.plot(epochs, history['val_acc'], marker='s', 
                   label=exp_name, linewidth=2, markersize=4)
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Validation Accuracy Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: all_training_curves.png")

def plot_individual_experiment(exp_name, history, output_dir):
    """
    Vẽ đồ thị chi tiết cho 1 experiment
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], marker='o', 
           label='Train Loss', linewidth=2, markersize=5, color='#2E86AB')
    ax.plot(epochs, history['val_loss'], marker='s', 
           label='Val Loss', linewidth=2, markersize=5, color='#A23B72')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'{exp_name} - Loss', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Accuracy
    ax = axes[1]
    ax.plot(epochs, history['train_acc'], marker='o', 
           label='Train Acc', linewidth=2, markersize=5, color='#2E86AB')
    ax.plot(epochs, history['val_acc'], marker='s', 
           label='Val Acc', linewidth=2, markersize=5, color='#A23B72')
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{exp_name} - Accuracy', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    safe_name = exp_name.replace('/', '_').replace(' ', '_')
    plt.savefig(os.path.join(output_dir, f'{safe_name}_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {safe_name}_curves.png")

def plot_test_metrics_comparison(experiments_results, output_dir):
    """
    Vẽ đồ thị so sánh metrics của tất cả experiments
    """
    exp_names = list(experiments_results.keys())
    metrics = ['test_accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    metric_labels = ['Accuracy (%)', 'Precision', 'Recall', 'F1 Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx]
        
        values = [experiments_results[exp][metric] if experiments_results[exp] else 0 
                 for exp in exp_names]
        
        bars = ax.bar(range(len(exp_names)), values, 
                     color=['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E'][:len(exp_names)],
                     edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(exp_names)))
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.set_title(f'Test {label} Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: test_metrics_comparison.png")

def plot_per_class_metrics(experiments_results, output_dir):
    """
    Vẽ đồ thị per-class metrics
    """
    for exp_name, results in experiments_results.items():
        if not results or 'per_class_metrics' not in results:
            continue
        
        per_class = results['per_class_metrics']
        class_names = list(per_class.keys())
        
        # Extract metrics
        precisions = [per_class[c]['precision'] for c in class_names]
        recalls = [per_class[c]['recall'] for c in class_names]
        f1s = [per_class[c]['f1'] for c in class_names]
        
        # Plot
        fig, ax = plt.subplots(figsize=(16, 8))
        
        x = np.arange(len(class_names))
        width = 0.25
        
        ax.bar(x - width, precisions, width, label='Precision', 
              color='#2E86AB', edgecolor='black')
        ax.bar(x, recalls, width, label='Recall', 
              color='#A23B72', edgecolor='black')
        ax.bar(x + width, f1s, width, label='F1 Score', 
              color='#F18F01', edgecolor='black')
        
        ax.set_xlabel('Class', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title(f'{exp_name} - Per-Class Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=9)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        safe_name = exp_name.replace('/', '_').replace(' ', '_')
        plt.savefig(os.path.join(output_dir, f'{safe_name}_per_class_metrics.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ Saved: {safe_name}_per_class_metrics.png")

def generate_summary_table(experiments_results, output_dir):
    """
    Tạo bảng tổng hợp kết quả
    """
    summary = []
    summary.append("="*100)
    summary.append("SUMMARY TABLE - TEST RESULTS")
    summary.append("="*100)
    summary.append(f"{'Experiment':<40} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    summary.append("-"*100)
    
    for exp_name, results in experiments_results.items():
        if results:
            summary.append(
                f"{exp_name:<40} "
                f"{results['test_accuracy']:>10.2f}% "
                f"{results['precision_weighted']:>11.4f} "
                f"{results['recall_weighted']:>11.4f} "
                f"{results['f1_weighted']:>11.4f}"
            )
    
    summary.append("="*100)
    
    # Save to file
    with open(os.path.join(output_dir, 'summary_table.txt'), 'w') as f:
        f.write('\n'.join(summary))
    
    # Print
    print('\n'.join(summary))

def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots')
    parser.add_argument('--experiments', type=str, nargs='+', default=None,
                       help='List of experiment names to visualize (default: all)')
    
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = os.path.join(args.results_dir, 'visualizations')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"{'='*80}")
    print("VISUALIZING RESULTS")
    print(f"{'='*80}")
    print(f"Results dir: {args.results_dir}")
    print(f"Output dir: {args.output_dir}")
    print()
    
    # Find all experiments
    if args.experiments:
        exp_names = args.experiments
    else:
        exp_names = [d for d in os.listdir(args.results_dir) 
                    if os.path.isdir(os.path.join(args.results_dir, d))]
    
    print(f"Found {len(exp_names)} experiments:")
    for exp in exp_names:
        print(f"  - {exp}")
    print()
    
    # Load data
    experiments_history = {}
    experiments_results = {}
    
    for exp_name in exp_names:
        exp_dir = os.path.join(args.results_dir, exp_name)
        
        history = load_history(exp_dir)
        results = load_test_results(exp_dir)
        
        if history:
            experiments_history[exp_name] = history
            print(f"✓ Loaded history: {exp_name}")
        
        if results:
            experiments_results[exp_name] = results
            print(f"✓ Loaded results: {exp_name}")
    
    print()
    
    # Generate plots
    if experiments_history:
        print("Generating training curves...")
        plot_training_curves(experiments_history, args.output_dir)
        
        for exp_name, history in experiments_history.items():
            plot_individual_experiment(exp_name, history, args.output_dir)
    
    if experiments_results:
        print("\nGenerating test metrics comparison...")
        plot_test_metrics_comparison(experiments_results, args.output_dir)
        
        print("\nGenerating per-class metrics...")
        plot_per_class_metrics(experiments_results, args.output_dir)
        
        print("\nGenerating summary table...")
        generate_summary_table(experiments_results, args.output_dir)
    
    print(f"\n{'='*80}")
    print("✓ VISUALIZATION COMPLETE")
    print(f"{'='*80}")
    print(f"Output saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
