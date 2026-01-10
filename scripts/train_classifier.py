#!/usr/bin/env python3
"""
Training script cho classification experiments
Support: ResNet18, ResNet50, ViT Tiny
"""
import os
import json
import argparse
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import timm
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class Trainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create output dir
        self.output_dir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Setup model
        self.model = self.create_model()
        self.model = self.model.to(self.device)
        
        # Setup data
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()
        self.num_classes = len(self.train_loader.dataset.classes)
        self.class_names = self.train_loader.dataset.classes
        
        # Setup training
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.create_optimizer()
        self.scheduler = self.create_scheduler()
        
        # Tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epoch_times': []
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        
    def create_model(self):
        """Tạo model theo architecture"""
        print(f"\nCreating model: {self.args.model}")
        
        if self.args.model == 'resnet18':
            model = models.resnet18(pretrained=self.args.pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.args.num_classes)
        
        elif self.args.model == 'resnet50':
            model = models.resnet50(pretrained=self.args.pretrained)
            model.fc = nn.Linear(model.fc.in_features, self.args.num_classes)
        
        elif self.args.model == 'vit_tiny':
            model = timm.create_model('vit_tiny_patch16_224', pretrained=self.args.pretrained)
            model.head = nn.Linear(model.head.in_features, self.args.num_classes)
        
        else:
            raise ValueError(f"Unknown model: {self.args.model}")
        
        print(f"✓ Model created with {self.args.num_classes} classes")
        return model
    
    def create_dataloaders(self):
        """Tạo dataloaders"""
        print("\nCreating dataloaders...")
        
        # Transforms
        if self.args.model == 'vit_tiny':
            img_size = 224
        else:
            img_size = 224
        
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        test_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Load datasets
        train_dataset = datasets.ImageFolder(self.args.train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(self.args.val_dir, transform=test_transform)
        test_dataset = datasets.ImageFolder(self.args.test_dir, transform=test_transform)
        
        print(f"  Train: {len(train_dataset)} images")
        print(f"  Val: {len(val_dataset)} images")
        print(f"  Test: {len(test_dataset)} images")
        print(f"  Classes: {len(train_dataset.classes)}")
        
        # Create loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def create_optimizer(self):
        """Tạo optimizer"""
        if self.args.optimizer == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.args.lr, 
                            weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.args.lr, 
                           momentum=0.9, weight_decay=self.args.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {self.args.optimizer}")
    
    def create_scheduler(self):
        """Tạo learning rate scheduler"""
        if self.args.scheduler == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.epochs
            )
        elif self.args.scheduler == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.1
            )
        elif self.args.scheduler == 'none':
            return None
        else:
            raise ValueError(f"Unknown scheduler: {self.args.scheduler}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.epochs} [Train]")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader, desc="Val"):
        """Validate/Test"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=desc)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{running_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_labels
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*80}")
        print("STARTING TRAINING")
        print(f"{'='*80}")
        print(f"Experiment: {self.args.exp_name}")
        print(f"Model: {self.args.model}")
        print(f"Epochs: {self.args.epochs}")
        print(f"Batch size: {self.args.batch_size}")
        print(f"Learning rate: {self.args.lr}")
        print(f"Optimizer: {self.args.optimizer}")
        print(f"Scheduler: {self.args.scheduler}")
        print()
        
        start_time = time.time()
        
        for epoch in range(self.args.epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_acc, _, _ = self.validate(self.val_loader, "Val")
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['epoch_times'].append(epoch_time)
            
            # Print summary
            print(f"\nEpoch {epoch+1}/{self.args.epochs} - {epoch_time:.2f}s")
            print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
            print(f"  Val:   Loss={val_loss:.4f}, Acc={val_acc:.2f}%")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pth')
                print(f"  ✓ New best model! Val Acc: {val_acc:.2f}%")
            
            # Save checkpoint periodically
            if (epoch + 1) % self.args.save_freq == 0:
                self.save_checkpoint(f'checkpoint_epoch{epoch+1}.pth')
        
        total_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best Val Acc: {self.best_val_acc:.2f}% at epoch {self.best_epoch+1}")
        
        # Save history
        self.save_history()
        
        return self.history
    
    def test(self):
        """Test on test set"""
        print(f"\n{'='*80}")
        print("TESTING ON TEST SET")
        print(f"{'='*80}")
        
        # Load best model
        best_model_path = os.path.join(self.output_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path}")
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Test
        test_loss, test_acc, test_preds, test_labels = self.validate(
            self.test_loader, "Test"
        )
        
        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, test_preds, average='weighted'
        )
        
        precision_per_class, recall_per_class, f1_per_class, support = \
            precision_recall_fscore_support(test_labels, test_preds, average=None)
        
        # Confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        
        # Print results
        print(f"\n{'='*80}")
        print("TEST RESULTS")
        print(f"{'='*80}")
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Precision (weighted): {precision:.4f}")
        print(f"Recall (weighted): {recall:.4f}")
        print(f"F1 Score (weighted): {f1:.4f}")
        print()
        
        # Per-class metrics
        print("Per-class metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {class_name}:")
            print(f"    Precision: {precision_per_class[i]:.4f}")
            print(f"    Recall: {recall_per_class[i]:.4f}")
            print(f"    F1: {f1_per_class[i]:.4f}")
            print(f"    Support: {support[i]}")
        
        # Save results
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'per_class_metrics': {}
        }
        
        for i, class_name in enumerate(self.class_names):
            results['per_class_metrics'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1': float(f1_per_class[i]),
                'support': int(support[i])
            }
        
        # Save results
        with open(os.path.join(self.output_dir, 'test_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        # Plot confusion matrix
        self.plot_confusion_matrix(cm, self.class_names)
        
        return results, cm
    
    def save_checkpoint(self, filename):
        """Save checkpoint"""
        checkpoint = {
            'epoch': len(self.history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'args': vars(self.args)
        }
        torch.save(checkpoint, os.path.join(self.output_dir, filename))
    
    def save_history(self):
        """Save training history"""
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.args.exp_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print(f"✓ Confusion matrix saved")

def main():
    parser = argparse.ArgumentParser(description='Train classifier')
    
    # Data
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--test_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/workspace/experiments_results')
    parser.add_argument('--exp_name', type=str, required=True)
    
    # Model
    parser.add_argument('--model', type=str, required=True,
                       choices=['resnet18', 'resnet50', 'vit_tiny'])
    parser.add_argument('--num_classes', type=int, default=15)
    parser.add_argument('--pretrained', action='store_true')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--optimizer', type=str, default='adam', 
                       choices=['adam', 'sgd'])
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'none'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=10)
    
    args = parser.parse_args()
    
    # Create trainer and train
    trainer = Trainer(args)
    trainer.train()
    trainer.test()
    
    print("\n✓ Training and testing complete!")

if __name__ == '__main__':
    main()