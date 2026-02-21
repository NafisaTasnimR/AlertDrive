"""
Sequence Dataset for CNN-LSTM Training

Loads sequences of images for temporal modeling.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
from collections import defaultdict


class SequenceDataset(Dataset):
    """
    Dataset that loads sequences of images for temporal modeling.
    
    Groups consecutive frames from the same video/subject into sequences.
    """
    
    def __init__(
        self, 
        csv_file, 
        root_dir, 
        sequence_length=10,
        transform=None,
        overlap=0.5,
        min_frames_per_video=None
    ):
        """
        Args:
            csv_file (str): Path to CSV file with columns: path, label
            root_dir (str): Root directory for images
            sequence_length (int): Number of frames per sequence
            transform (callable): Optional transform to apply to images
            overlap (float): Overlap ratio between consecutive sequences (0-1)
            min_frames_per_video (int): Minimum frames required to include a video
        """
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.overlap = overlap
        
        # Load CSV
        self.data = pd.read_csv(csv_file)
        
        # Group images by video/subject
        self.sequences = self._create_sequences()
        
        # Filter by minimum frames if specified
        if min_frames_per_video:
            self.sequences = [s for s in self.sequences if len(s['frames']) >= min_frames_per_video]
        
        print(f"Created {len(self.sequences)} sequences from {len(self.data)} images")
        print(f"Sequence length: {sequence_length}, Overlap: {overlap}")
    
    def _extract_video_id(self, path):
        """
        Extract video/subject identifier from path.
        
        Assumes format: category/subject_glasses_action_frame_label.jpg
        Example: drowsy/005_noglasses_sleepyCombination_2964_drowsy.jpg
        Returns: "005_noglasses_sleepyCombination_drowsy"
        """
        filename = os.path.basename(path)
        parts = filename.split('_')
        
        if len(parts) >= 4:
            # Extract subject, glasses status, and action
            subject = parts[0]
            glasses = parts[1]
            action = parts[2]
            label = parts[-1].replace('.jpg', '')
            
            video_id = f"{subject}_{glasses}_{action}_{label}"
            return video_id
        
        return filename  # fallback
    
    def _extract_frame_number(self, path):
        """
        Extract frame number from filename.
        
        Example: 005_noglasses_sleepyCombination_2964_drowsy.jpg -> 2964
        """
        filename = os.path.basename(path)
        parts = filename.split('_')
        
        if len(parts) >= 4:
            try:
                frame_num = int(parts[-2])
                return frame_num
            except ValueError:
                pass
        
        return 0  # fallback
    
    def _create_sequences(self):
        """
        Group images into sequences based on video ID and frame order.
        """
        # Group by video ID
        video_groups = defaultdict(list)
        
        for idx, row in self.data.iterrows():
            path = row['path']
            label = row['label']
            
            video_id = self._extract_video_id(path)
            frame_num = self._extract_frame_number(path)
            
            video_groups[video_id].append({
                'path': path,
                'label': label,
                'frame': frame_num
            })
        
        # Sort frames within each video and create sequences
        sequences = []
        
        for video_id, frames in video_groups.items():
            # Sort by frame number
            frames = sorted(frames, key=lambda x: x['frame'])
            
            # Skip if not enough frames
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
                    'video_id': video_id,
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
                blank = torch.zeros((3, 224, 224))
                images.append(blank)
        
        # Stack images into sequence
        images = torch.stack(images)  # (seq_len, C, H, W)
        
        return images, label


def get_transforms(split='train', image_size=224):
    """
    Get appropriate transforms for train/val/test splits.
    
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
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
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


def create_dataloaders(
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
    Create train, validation, and test dataloaders.
    
    Args:
        train_csv (str): Path to training CSV
        val_csv (str): Path to validation CSV
        test_csv (str): Path to test CSV
        root_dir (str): Root directory for images
        batch_size (int): Batch size
        sequence_length (int): Number of frames per sequence
        overlap (float): Overlap between sequences
        num_workers (int): Number of worker processes
        image_size (int): Target image size
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = SequenceDataset(
        csv_file=train_csv,
        root_dir=root_dir,
        sequence_length=sequence_length,
        transform=get_transforms('train', image_size),
        overlap=overlap
    )
    
    val_dataset = SequenceDataset(
        csv_file=val_csv,
        root_dir=root_dir,
        sequence_length=sequence_length,
        transform=get_transforms('val', image_size),
        overlap=0  # No overlap for validation
    )
    
    test_dataset = SequenceDataset(
        csv_file=test_csv,
        root_dir=root_dir,
        sequence_length=sequence_length,
        transform=get_transforms('test', image_size),
        overlap=0  # No overlap for testing
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
    
    # Print dataset info
    print(f"\nDataset Statistics:")
    print(f"  Training sequences: {len(train_dataset)}")
    print(f"  Validation sequences: {len(val_dataset)}")
    print(f"  Test sequences: {len(test_dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {sequence_length}")
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test the dataset
    print("Testing SequenceDataset...")
    
    root_dir = "E:/varsity/Semester-5/CSE_4554_ML_Lab/project/AlertDrive"
    train_csv = "splits/train.csv"
    
    dataset = SequenceDataset(
        csv_file=os.path.join(root_dir, train_csv),
        root_dir=root_dir,
        sequence_length=10,
        transform=get_transforms('train'),
        overlap=0.5
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Test getting a sample
    images, label = dataset[0]
    print(f"Sample batch shape: {images.shape}")
    print(f"Label: {label}")
    
    print("\n✓ Dataset test passed!")
