import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModelForImageClassification

class HFCNNResNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(
            "microsoft/resnet-50"
        )

        self.model = AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-50",
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        

    def forward(self, x):
        return self.model(pixel_values=x).logits