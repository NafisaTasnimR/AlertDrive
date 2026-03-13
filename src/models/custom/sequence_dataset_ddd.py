"""
Sequence Dataset for DDD CNN-LSTM Training

Loads sequences of images from the Driver Drowsiness Dataset (DDD) for temporal modeling.
Unlike the multi-class dataset, DDD images have simpler naming conventions.
"""

import os
import re
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from collections import defaultdict


class AddGaussianNoise:
    """Add Gaussian noise to tensor"""
    
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std
    
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean


class SequenceDatasetDDD(Dataset):
    """
    Dataset that loads sequences of images from DDD for temporal modeling.
    
    DDD images have simple sequential naming (e.g., A0013.png, a0002.png).
    Sequences are created by grouping consecutive frames within each class.
    """
    
    def __init__(
        self, 
        csv_file, 
        root_dir, 
        sequence_length=10,
        transform=None,
        overlap=0.5,
        min_frames_per_group=None
    ):
        """
        Args:
            csv_file (str): Path to CSV file with columns: path, label
            root_dir (str): Root directory for images (e.g., 'data/processed_ddd')
            sequence_length (int): Number of frames per sequence
            transform (callable): Optional transform to apply to images
            overlap (float): Overlap ratio between consecutive sequences (0-1)
            min_frames_per_group (int): Minimum frames required to include a group
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.overlap = overlap
        
        # Load CSV
        self.data = pd.read_csv(csv_file)
        
        # Create sequences
        self.sequences = self._create_sequences()
        
        # Filter by minimum frames if specified
        if min_frames_per_group:
            self.sequences = [s for s in self.sequences 
                            if len(s['frames']) >= min_frames_per_group]
        
        print(f"Created {len(self.sequences)} sequences from {len(self.data)} images")
        print(f"Sequence length: {sequence_length}, Overlap: {overlap}")
    
    def _extract_frame_info(self, path):
        """
        Extract frame information from DDD image path.
        
        DDD naming convention:
        - Drowsy: A0013.png, A0014.png, ... (capital A)
        - Non Drowsy: a0002.png, a0003.png, ... (lowercase a)
        
        Args:
            path (str): Image path (e.g., 'drowsy/A0013.png')
        
        Returns:
            tuple: (group_id, frame_number)
                - group_id: 'drowsy_A' or 'notdrowsy_a' etc.
                - frame_number: Integer frame ID
        """
        filename = os.path.basename(path)
        basename, _ = os.path.splitext(filename)

        # Normalize class path to keep group IDs stable across platforms.
        class_name = os.path.dirname(path).replace('\\', '/').strip('/')

        # Case 1: Underscore naming (preferred): remove numeric frame token and
        # keep the remaining tokens as the sequence/video identifier.
        if '_' in basename:
            tokens = basename.split('_')
            frame_idx = None

            for i in range(len(tokens) - 1, -1, -1):
                if tokens[i].isdigit():
                    frame_idx = i
                    break

            if frame_idx is not None:
                frame_number = int(tokens[frame_idx])
                group_tokens = [t for j, t in enumerate(tokens) if j != frame_idx]
                group_stem = '_'.join(group_tokens) if group_tokens else basename
                return f"{class_name}/{group_stem}", frame_number

        # Case 2: Compact naming like A0013.png / a0002.png.
        match = re.match(r'^(.*?)(\d+)$', basename)
        if match:
            prefix = match.group(1) if match.group(1) else 'unknown'
            frame_number = int(match.group(2))
            return f"{class_name}/{prefix}", frame_number

        # Fallback: treat whole basename as one group with frame 0.
        return f"{class_name}/{basename}", 0
    
    def _create_sequences(self):
        """
        Group images into sequences based on group ID and frame order.
        
        For DDD, we group by:
        1. Class (drowsy/notdrowsy)
        2. Filename prefix (A, a, etc.)
        3. Sequential frame numbers
        """
        # Group by video/sequence ID
        frame_groups = defaultdict(list)
        
        for idx, row in self.data.iterrows():
            path = row['path']
            label = row['label']
            
            group_id, frame_num = self._extract_frame_info(path)
            
            frame_groups[group_id].append({
                'path': path,
                'label': label,
                'frame': frame_num
            })
        
        # Sort frames within each group and create sequences
        sequences = []
        
        for group_id, frames in frame_groups.items():
            # Sort by frame number
            frames = sorted(frames, key=lambda x: x['frame'])
            
            # Skip short clips instead of duplicating frames. Repeating frames
            # creates synthetic easy patterns that hurt generalization.
            if len(frames) < self.sequence_length:
                continue
            
            # Create overlapping sequences
            step_size = max(1, int(self.sequence_length * (1 - self.overlap)))
            
            for start_idx in range(0, len(frames) - self.sequence_length + 1, step_size):
                sequence_frames = frames[start_idx:start_idx + self.sequence_length]
                
                # Use majority label for the sequence
                labels = [f['label'] for f in sequence_frames]
                majority_label = max(set(labels), key=labels.count)
                
                sequences.append({
                    'group_id': group_id,
                    'frames': sequence_frames,
                    'label': majority_label
                })
        
        return sequences
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a sequence of images and corresponding label.
        
        Returns:
            images (torch.Tensor): Tensor of shape (seq_len, C, H, W)
            label (int): Label for the sequence
        """
        sequence = self.sequences[idx]
        frames = sequence['frames']
        label = sequence['label']
        
        # Load and transform images
        images = []
        for frame in frames:
            img_path = os.path.join(self.root_dir, frame['path'])
            
            try:
                image = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    image = self.transform(image)
                
                images.append(image)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                # Use blank image as fallback
                if self.transform:
                    blank = torch.zeros((3, 224, 224))
                else:
                    blank = Image.new('RGB', (224, 224), (0, 0, 0))
                    if self.transform:
                        blank = self.transform(blank)
                images.append(blank)
        
        # Stack images into sequence
        images = torch.stack(images)  # (seq_len, C, H, W)
        
        return images, label


def get_transforms_ddd(split='train', image_size=224):
    """
    Get appropriate transforms for DDD train/val/test splits.
    
    Args:
        split (str): 'train', 'val', or 'test'
        image_size (int): Target image size
    
    Returns:
        torchvision.transforms.Compose: Composed transforms
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.15)),
            AddGaussianNoise(mean=0., std=0.05),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_dataloaders_ddd(
    train_csv,
    val_csv,
    test_csv,
    root_dir,
    batch_size=8,
    sequence_length=10,
    overlap=0.5,
    num_workers=4,
    image_size=224
):
    """
    Create train, validation, and test dataloaders for DDD.
    
    Args:
        train_csv (str): Path to training CSV (e.g., 'splits/ddd_train.csv')
        val_csv (str): Path to validation CSV (e.g., 'splits/ddd_val.csv')
        test_csv (str): Path to test CSV (e.g., 'splits/ddd_test.csv')
        root_dir (str): Root directory for images (e.g., 'data/processed_ddd')
        batch_size (int): Batch size
        sequence_length (int): Number of frames per sequence
        overlap (float): Overlap between sequences
        num_workers (int): Number of worker processes
        image_size (int): Target image size
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SequenceDatasetDDD(
        csv_file=train_csv,
        root_dir=root_dir,
        sequence_length=sequence_length,
        transform=get_transforms_ddd('train', image_size),
        overlap=overlap
    )
    
    val_dataset = SequenceDatasetDDD(
        csv_file=val_csv,
        root_dir=root_dir,
        sequence_length=sequence_length,
        transform=get_transforms_ddd('val', image_size),
        overlap=0
    )
    
    test_dataset = SequenceDatasetDDD(
        csv_file=test_csv,
        root_dir=root_dir,
        sequence_length=sequence_length,
        transform=get_transforms_ddd('test', image_size),
        overlap=0
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataloaders created:")
    print(f"  Train: {len(train_dataset)} sequences, {len(train_loader)} batches")
    print(f"  Val:   {len(val_dataset)} sequences, {len(val_loader)} batches")
    print(f"  Test:  {len(test_dataset)} sequences, {len(test_loader)} batches")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing DDD Sequence Dataset...")
    
    # Create a test dataset
    dataset = SequenceDatasetDDD(
        csv_file='splits/ddd_train.csv',
        root_dir='data/processed_ddd',
        sequence_length=10,
        transform=get_transforms_ddd('train'),
        overlap=0.5
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Get a sample
    if len(dataset) > 0:
        images, label = dataset[0]
        print(f"\nSample batch:")
        print(f"  Images shape: {images.shape}")  # Should be (10, 3, 224, 224)
        print(f"  Label: {label}")
