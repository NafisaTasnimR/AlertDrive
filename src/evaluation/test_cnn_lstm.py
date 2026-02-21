"""
Evaluation Script for CNN-LSTM Model

Test and evaluate the trained CNN-LSTM model.
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.custom.cnn_lstm import create_cnn_lstm
from src.models.custom.sequence_dataset import create_dataloaders


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.show()


def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to: {save_path}")
    
    plt.show()


def evaluate_model(
    model,
    dataloader,
    device,
    class_names=['Not Drowsy', 'Drowsy']
):
    """
    Evaluate model on a dataset.
    
    Returns:
        Dict with evaluation metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    print("\nEvaluating model...")
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of drowsy class
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    # ROC-AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
    except:
        auc = None
        fpr, tpr = None, None
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Classification report
    report = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names,
        digits=4
    )
    
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'confusion_matrix': cm,
        'classification_report': report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'fpr': fpr,
        'tpr': tpr
    }
    
    return results


def test_cnn_lstm(
    model_path,
    test_csv='splits/test.csv',
    root_dir='.',
    batch_size=8,
    sequence_length=10,
    num_workers=4,
    save_results=True,
    results_dir='results/cnn_lstm'
):
    """
    Test CNN-LSTM model on test set.
    
    Args:
        model_path (str): Path to trained model checkpoint
        test_csv (str): Path to test CSV
        root_dir (str): Root directory for images
        batch_size (int): Batch size for testing
        sequence_length (int): Sequence length
        num_workers (int): Number of workers
        save_results (bool): Whether to save results
        results_dir (str): Directory to save results
    """
    
    print(f"\n{'='*80}")
    print(f"CNN-LSTM MODEL EVALUATION")
    print(f"{'='*80}")
    print(f"Model: {model_path}")
    print(f"{'='*80}\n")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    # Load checkpoint
    print("[1/5] Loading model checkpoint...")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get model config
    config = checkpoint.get('config', {})
    num_classes = config.get('num_classes', 2)
    hidden_size = config.get('hidden_size', 128)
    num_layers = config.get('num_layers', 1)
    dropout = config.get('dropout', 0.3)
    seq_len = config.get('sequence_length', sequence_length)
    
    print(f"Model config:")
    print(f"  Classes: {num_classes}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  LSTM layers: {num_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Sequence length: {seq_len}")
    
    # Create model
    print("\n[2/5] Creating model...")
    model = create_cnn_lstm(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully!")
    
    # Create test dataloader
    print("\n[3/5] Loading test data...")
    from src.models.custom.sequence_dataset import SequenceDataset, get_transforms
    
    test_dataset = SequenceDataset(
        csv_file=test_csv,
        root_dir=root_dir,
        sequence_length=seq_len,
        transform=get_transforms('test'),
        overlap=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Test sequences: {len(test_dataset)}")
    
    # Evaluate
    print("\n[4/5] Evaluating on test set...")
    results = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device
    )
    
    # Print results
    print("\n" + "="*80)
    print("TEST RESULTS")
    print("="*80)
    print(f"Accuracy:  {results['accuracy']*100:.2f}%")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    if results['auc']:
        print(f"ROC-AUC:   {results['auc']:.4f}")
    print("="*80)
    
    print("\nClassification Report:")
    print(results['classification_report'])
    
    print("\nConfusion Matrix:")
    print(results['confusion_matrix'])
    
    # Save results
    if save_results:
        print(f"\n[5/5] Saving results to {results_dir}...")
        os.makedirs(results_dir, exist_ok=True)
        
        # Extract model name from path
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        # Save metrics as JSON
        metrics = {
            'accuracy': float(results['accuracy']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'auc': float(results['auc']) if results['auc'] else None,
            'confusion_matrix': results['confusion_matrix'].tolist()
        }
        
        metrics_path = os.path.join(results_dir, f'{model_name}_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to: {metrics_path}")
        
        # Save confusion matrix plot
        cm_path = os.path.join(results_dir, f'{model_name}_confusion_matrix.png')
        plot_confusion_matrix(
            results['confusion_matrix'],
            class_names=['Not Drowsy', 'Drowsy'],
            save_path=cm_path
        )
        
        # Save ROC curve if available
        if results['auc'] and results['fpr'] is not None:
            roc_path = os.path.join(results_dir, f'{model_name}_roc_curve.png')
            plot_roc_curve(
                results['fpr'],
                results['tpr'],
                results['auc'],
                save_path=roc_path
            )
        
        # Save classification report
        report_path = os.path.join(results_dir, f'{model_name}_classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(results['classification_report'])
        print(f"Classification report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate CNN-LSTM model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test_csv', type=str, default='splits/test.csv',
                       help='Path to test CSV')
    parser.add_argument('--root_dir', type=str, default='.',
                       help='Root directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--sequence_length', type=int, default=10,
                       help='Sequence length')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers')
    parser.add_argument('--results_dir', type=str, default='results/cnn_lstm',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run evaluation
    results = test_cnn_lstm(
        model_path=args.model,
        test_csv=args.test_csv,
        root_dir=args.root_dir,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        num_workers=args.num_workers,
        save_results=True,
        results_dir=args.results_dir
    )
