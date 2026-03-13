"""
Test Script for CNN-LSTM Model on DDD Dataset

Evaluate the trained CNN-LSTM model on the DDD test set.
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.custom.cnn_lstm import CNN_LSTM
from src.models.custom.sequence_dataset_ddd import SequenceDatasetDDD, get_transforms_ddd


def load_model(checkpoint_path, device='cuda'):
    """
    Load trained CNN-LSTM model from checkpoint.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        device (str): Device to load model on
    
    Returns:
        CNN_LSTM: Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model config
    config = checkpoint['config']
    
    # Create model
    model = CNN_LSTM(
        num_classes=config.get('num_classes', 2),
        hidden_size=config.get('hidden_size', 128),
        num_layers=config.get('num_layers', 1),
        dropout=config.get('dropout', 0.3)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Val Accuracy: {checkpoint.get('val_acc', 'N/A'):.2f}%")
    
    return model, config


def evaluate_model(
    model,
    test_loader,
    device='cuda',
    save_dir='results'
):
    """
    Evaluate model on test set.
    
    Args:
        model: Trained CNN-LSTM model
        test_loader: Test data loader
        device: Device to run evaluation on
        save_dir: Directory to save results
    
    Returns:
        dict: Evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model on test set...")
    
    with torch.no_grad():
        for sequences, labels in tqdm(test_loader, desc="Testing"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    
    print("\n" + "="*80)
    print("TEST RESULTS - DDD DATASET")
    print("="*80)
    print(f"\nOverall Accuracy: {accuracy*100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(
        all_labels, 
        all_preds,
        target_names=['Not Drowsy', 'Drowsy'],
        digits=4
    ))
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Per-class accuracy
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    print("\nPer-Class Accuracy:")
    print(f"  Not Drowsy: {class_accuracies[0]*100:.2f}%")
    print(f"  Drowsy:     {class_accuracies[1]*100:.2f}%")
    
    # Save confusion matrix plot
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Drowsy', 'Drowsy'],
                yticklabels=['Not Drowsy', 'Drowsy'])
    plt.title('Confusion Matrix - DDD Dataset')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(save_dir, 'confusion_matrix_ddd.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {cm_path}")
    plt.close()
    
    # Save results
    results = {
        'accuracy': float(accuracy),
        'per_class_accuracy': {
            'not_drowsy': float(class_accuracies[0]),
            'drowsy': float(class_accuracies[1])
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(
            all_labels, 
            all_preds,
            target_names=['Not Drowsy', 'Drowsy'],
            output_dict=True
        )
    }
    
    results_path = os.path.join(save_dir, 'test_results_ddd.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to: {results_path}")
    print("="*80 + "\n")
    
    return results


def test_cnn_lstm_ddd(
    checkpoint_path,
    test_csv='splits/ddd_test.csv',
    root_dir='data/processed_ddd',
    batch_size=8,
    sequence_length=10,
    overlap=0.5,
    num_workers=4,
    save_dir='results'
):
    """
    Test CNN-LSTM model on DDD dataset.
    
    Args:
        checkpoint_path (str): Path to model checkpoint
        test_csv (str): Path to test CSV
        root_dir (str): Root directory for images
        batch_size (int): Batch size for testing
        sequence_length (int): Sequence length used during training
        overlap (float): Sequence overlap
        num_workers (int): Number of data loading workers
        save_dir (str): Directory to save results
    
    Returns:
        dict: Test results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*80)
    print("CNN-LSTM TESTING ON DDD DATASET")
    print("="*80)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test CSV: {test_csv}")
    print("="*80 + "\n")
    
    # Load model
    print("[1/3] Loading model...")
    model, config = load_model(checkpoint_path, device)
    
    # Update sequence_length from config if available
    if 'sequence_length' in config:
        sequence_length = config['sequence_length']
        print(f"Using sequence length from config: {sequence_length}")
    
    # Create test dataloader
    print("\n[2/3] Loading test data...")
    test_dataset = SequenceDatasetDDD(
        csv_file=test_csv,
        root_dir=root_dir,
        sequence_length=sequence_length,
        transform=get_transforms_ddd('test'),
        overlap=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Evaluate
    print("\n[3/3] Running evaluation...")
    results = evaluate_model(model, test_loader, device, save_dir)
    
    print("\n✓ Testing completed successfully!")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test CNN-LSTM on DDD dataset')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, default='splits/ddd_test.csv',
                       help='Path to test CSV')
    parser.add_argument('--root_dir', type=str, default='data/processed_ddd',
                       help='Root directory for images')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Sequence length')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Sequence overlap')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--save_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run testing
    results = test_cnn_lstm_ddd(
        checkpoint_path=args.checkpoint,
        test_csv=args.test_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        overlap=args.overlap,
        num_workers=args.num_workers,
        save_dir=args.save_dir
    )
