"""
Training Script for CNN-LSTM Model

Train the CNN-LSTM model for drowsy driver detection using sequential images.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.custom.cnn_lstm import create_cnn_lstm
from src.models.custom.sequence_dataset import create_dataloaders


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights:
                    model.load_state_dict(self.best_weights)
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.best_weights = model.state_dict().copy()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (sequences, labels) in enumerate(pbar):
        # Move to device
        sequences = sequences.to(device)  # (batch, seq_len, C, H, W)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device, epoch):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    with torch.no_grad():
        for batch_idx, (sequences, labels) in enumerate(pbar):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train_cnn_lstm(
    # Data parameters
    train_csv='splits/train.csv',
    val_csv='splits/val.csv',
    test_csv='splits/test.csv',
    root_dir='.',
    
    # Model parameters
    num_classes=2,
    hidden_size=128,
    num_layers=1,
    dropout=0.3,
    freeze_cnn=False,
    
    # Sequence parameters
    sequence_length=10,
    overlap=0.5,
    
    # Training parameters
    batch_size=8,
    num_epochs=50,
    learning_rate=0.001,
    weight_decay=1e-5,
    
    # Other parameters
    num_workers=4,
    patience=10,
    save_dir='trained_models/custom',
    log_dir='logs/cnn_lstm'
):
    """
    Train CNN-LSTM model for drowsy driver detection.
    """
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*80}")
    print(f"CNN-LSTM TRAINING")
    print(f"{'='*80}")
    print(f"Device: {device}")
    print(f"Sequence length: {sequence_length}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*80}\n")
    
    # Create directories
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Tensorboard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f"{log_dir}/run_{timestamp}")
    
    # Create dataloaders
    print("\n[1/6] Loading data...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=test_csv,
        root_dir=root_dir,
        batch_size=batch_size,
        sequence_length=sequence_length,
        overlap=overlap,
        num_workers=num_workers
    )
    
    # Create model
    print("\n[2/6] Creating model...")
    model = create_cnn_lstm(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        freeze_cnn=freeze_cnn,
        device=device
    )
    
    # Loss and optimizer
    print("\n[3/6] Setting up training...")
    
    # Calculate class weights for balanced training if needed
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=patience, restore_best_weights=True)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_model_path = None
    
    # Training loop
    print(f"\n[4/6] Training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Tensorboard logging
        writer.add_scalars('Loss', {
            'train': train_loss,
            'val': val_loss
        }, epoch)
        writer.add_scalars('Accuracy', {
            'train': train_acc,
            'val': val_acc
        }, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(save_dir, f'cnn_lstm_best_{timestamp}.pth')
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': {
                    'num_classes': num_classes,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': dropout,
                    'sequence_length': sequence_length
                }
            }, best_model_path)
            
            print(f"  ✓ Best model saved! (Val Acc: {val_acc:.2f}%)")
        
        # Early stopping check
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    training_time = time.time() - start_time
    
    # Save final model
    print("\n[5/6] Saving final model...")
    final_model_path = os.path.join(save_dir, f'cnn_lstm_final_{timestamp}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'config': {
            'num_classes': num_classes,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': dropout,
            'sequence_length': sequence_length
        }
    }, final_model_path)
    
    # Save training history
    history_path = os.path.join(save_dir, f'training_history_{timestamp}.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    
    print(f"Final model saved to: {final_model_path}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Training history saved to: {history_path}")
    
    # Final evaluation on test set
    print("\n[6/6] Evaluating on test set...")
    test_loss, test_acc, test_preds, test_labels = validate(
        model, test_loader, criterion, device, "Test"
    )
    
    print(f"\nTest Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Acc:  {test_acc:.2f}%")
    
    # Training summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Total training time: {training_time/60:.2f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Final test accuracy: {test_acc:.2f}%")
    print(f"Best model: {best_model_path}")
    print(f"{'='*80}\n")
    
    writer.close()
    
    return model, history, best_model_path


if __name__ == "__main__":
    # Default configuration
    config = {
        'train_csv': 'splits/train.csv',
        'val_csv': 'splits/val.csv',
        'test_csv': 'splits/test.csv',
        'root_dir': '.',
        
        # Model config
        'num_classes': 2,
        'hidden_size': 128,
        'num_layers': 1,
        'dropout': 0.3,
        'freeze_cnn': False,
        
        # Sequence config
        'sequence_length': 10,
        'overlap': 0.5,
        
        # Training config
        'batch_size': 8,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'num_workers': 4,
        'patience': 10,
        
        # Save paths
        'save_dir': 'trained_models/custom',
        'log_dir': 'logs/cnn_lstm'
    }
    
    # Train model
    model, history, best_model_path = train_cnn_lstm(**config)
